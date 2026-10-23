// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
// Opt-in comparison: unchanged upstream WGSL on the same Vulkan GPU.
import fs from "node:fs";
import { createHash } from "node:crypto";
import path from "node:path";
import { packNative } from "./pack.mjs";
import { create, globals } from "webgpu";
Object.assign(globalThis, globals);
const filename = process.argv[2];
if (!filename)
  throw new Error("Usage: node benchmark.mjs export.json [iterations=30]");
const cfg = JSON.parse(fs.readFileSync(filename, "utf8"));
cfg.gpuWords ??= path.basename(filename).replace(/\.json$/, ".gpu.bin");
cfg.rootSize ??= cfg.root_size;
cfg.voxelSize ??= cfg.voxel_size;
cfg.inverseViewProjection ??= cfg.inverse_vp;
cfg.insideSolid ??= cfg.inside;
cfg.projScaleY ??= 1 / Math.tan(Math.PI / 6);
// Extra cells are empty padding. Upstream's block DDA requires whole bricks.
cfg.dimensions = cfg.dimensions.map((v) => Math.ceil(v / 4) * 4);
const base = path.dirname(path.resolve(filename));
const revision = "34c137ab1c010b99a1c199b68c36210990331262";
const sourceUrl = `https://raw.githubusercontent.com/playcanvas/supersplat-viewer/${revision}/src/voxel-debug-overlay.ts`;
let src;
if (process.env.SUPERSPLAT_OVERLAY_SOURCE) {
  src = fs.readFileSync(process.env.SUPERSPLAT_OVERLAY_SOURCE, "utf8");
} else {
  const response = await fetch(sourceUrl);
  if (!response.ok)
    throw new Error(`Upstream source fetch failed: ${response.status}`);
  src = await response.text();
}
if (
  createHash("sha256").update(src).digest("hex") !==
  "9a7636e27da247f31b789954d6a36f73e37c8565ee009f6be50037d9ec107b99"
)
  throw new Error("Upstream source does not match the pinned revision");
const wgsl = src.match(
  /const voxelOverlayWGSL = \/\* wgsl \*\/ `([\s\S]*?)\n`;/,
)[1];
const wordsFile = fs.readFileSync(path.resolve(base, cfg.gpuWords));
const words = new Uint32Array(
  wordsFile.buffer,
  wordsFile.byteOffset,
  wordsFile.byteLength / 4,
);
const rootSize = cfg.rootSize;
if (rootSize < 8 || rootSize & (rootSize - 1))
  throw new Error("rootSize must be power of two >=8");
const { packedNodes, packedLeaves, nodeCount, leafCount } = packNative(
  words,
  rootSize,
);
const width = cfg.width ?? 1920,
  height = cfg.height ?? 1080;
const uniformData = new ArrayBuffer(128),
  u = new DataView(uniformData);
const matrix = cfg.inverseViewProjection;
if (matrix.length !== 16)
  throw new Error("inverseViewProjection needs16 column-major floats");
for (let i = 0; i < 16; i++) u.setFloat32(i * 4, matrix[i], true);
u.setUint32(64, width, true);
u.setUint32(68, height, true);
for (let i = 0; i < 3; i++) u.setFloat32(72 + i * 4, cfg.origin[i], true);
u.setFloat32(84, cfg.voxelSize, true);
for (let i = 0; i < 3; i++) u.setUint32(88 + i * 4, cfg.dimensions[i], true);
if (cfg.dimensions.some((v) => v % 4 !== 0))
  throw new Error(
    "Upstream block DDA needs dimensions rounded up to multiples of4. Export same padded grid in native comparison.",
  );
u.setUint32(100, 4, true);
u.setUint32(104, Math.log2(rootSize) - 2, true);
u.setFloat32(108, cfg.projScaleY, true);
u.setUint32(112, 0, true);
u.setUint32(116, cfg.insideSolid ? 1 : 0, true);
const gpu = create([
  "backend=vulkan",
  "enable-dawn-features=allow_unsafe_apis",
  "disable-dawn-features=timestamp_quantization",
]);
const adapter = await gpu.requestAdapter({
  powerPreference: "high-performance",
});
if (!adapter) throw new Error("No Vulkan WebGPU adapter");
if (adapter.info.isFallbackAdapter)
  throw new Error("Software WebGPU adapter is not a valid GPU comparison");
if (!adapter.features.has("timestamp-query"))
  throw new Error("timestamp-query unsupported; cannot claim GPU timing");
const colorFormat = cfg.colorFormat ?? "rgba32float";
const features = ["timestamp-query"];
if (colorFormat === "rgba32float") {
  if (!adapter.features.has("float32-blendable"))
    throw new Error(
      "RGBA32F blending unsupported; select rgba8unorm explicitly and report attachment difference",
    );
  features.push("float32-blendable");
}
const maxBufferSize = Math.max(
  128 * 1024 * 1024,
  packedNodes.byteLength,
  packedLeaves.byteLength,
);
const device = await adapter.requestDevice({
  requiredFeatures: features,
  requiredLimits: {
    maxStorageBufferBindingSize: maxBufferSize,
    maxBufferSize: maxBufferSize,
  },
});
device.addEventListener("uncapturederror", (e) => {
  console.error(e.error);
  process.exitCode = 1;
});
function buffer(data, usage) {
  const b = device.createBuffer({
    size: Math.max(4, data.byteLength),
    usage: usage | GPUBufferUsage.COPY_DST,
  });
  if (data.byteLength) device.queue.writeBuffer(b, 0, data);
  return b;
}
const uniform = buffer(uniformData, GPUBufferUsage.UNIFORM),
  nb = buffer(packedNodes, GPUBufferUsage.STORAGE),
  lb = buffer(packedLeaves, GPUBufferUsage.STORAGE);
const output = device.createTexture({
  size: [width, height],
  format: "rgba8unorm",
  usage:
    GPUTextureUsage.STORAGE_BINDING |
    GPUTextureUsage.TEXTURE_BINDING |
    GPUTextureUsage.COPY_SRC,
});
const target = device.createTexture({
  size: [width, height],
  format: colorFormat,
  usage: GPUTextureUsage.RENDER_ATTACHMENT,
});
const depth = device.createTexture({
  size: [width, height],
  format: "depth32float",
  usage: GPUTextureUsage.RENDER_ATTACHMENT,
});
const module = device.createShaderModule({ code: wgsl });
const info = await module.getCompilationInfo();
for (const m of info.messages)
  if (m.type === "error") throw new Error(m.message);
const pipeline = await device.createComputePipelineAsync({
  layout: "auto",
  compute: { module, entryPoint: "main" },
});
const bindings = device.createBindGroup({
  layout: pipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: uniform } },
    { binding: 1, resource: { buffer: nb } },
    { binding: 2, resource: { buffer: lb } },
    { binding: 3, resource: output.createView() },
  ],
});
const quadModule = device.createShaderModule({
  code: `
struct V { @builtin(position) p: vec4f, @location(0) uv: vec2f };
@vertex fn vs(@builtin(vertex_index) index:u32)->V {
 var p=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));var o:V;
 o.p=vec4f(p[index],0,1);o.uv=vec2f(p[index].x*.5+.5,.5-p[index].y*.5);return o;
}
@group(0)@binding(0)var img:texture_2d<f32>;
@group(0)@binding(1)var smp:sampler;
@fragment fn fs(@location(0)uv:vec2f)->@location(0)vec4f{return textureSample(img,smp,uv);}
`,
});
const blend = {
  color: {
    operation: "add",
    srcFactor: "one",
    dstFactor: "one-minus-src-alpha",
  },
  alpha: {
    operation: "add",
    srcFactor: "one",
    dstFactor: "one-minus-src-alpha",
  },
};
const quad = await device.createRenderPipelineAsync({
  layout: "auto",
  vertex: { module: quadModule, entryPoint: "vs" },
  fragment: {
    module: quadModule,
    entryPoint: "fs",
    targets: [{ format: colorFormat, blend }],
  },
  depthStencil: {
    format: "depth32float",
    depthWriteEnabled: false,
    depthCompare: "always",
  },
  primitive: { topology: "triangle-list" },
});
const qb = device.createBindGroup({
  layout: quad.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: output.createView() },
    {
      binding: 1,
      resource: device.createSampler({
        magFilter: "linear",
        minFilter: "linear",
      }),
    },
  ],
});
const queries = device.createQuerySet({ type: "timestamp", count: 4 });
const resolved = device.createBuffer({
  size: 32,
  usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
});
const readback = device.createBuffer({
  size: 32,
  usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
});
const computeTimes = [],
  compositeTimes = [],
  totalTimes = [];
for (let i = -10; i < Number(process.argv[3] ?? 30); i++) {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass({
    timestampWrites: {
      querySet: queries,
      beginningOfPassWriteIndex: 0,
      endOfPassWriteIndex: 1,
    },
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindings);
  pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
  pass.end();
  const render = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: target.createView(),
        loadOp: "clear",
        storeOp: "store",
        clearValue: [0, 0, 0, 0],
      },
    ],
    depthStencilAttachment: {
      view: depth.createView(),
      depthClearValue: 1,
      depthLoadOp: "clear",
      depthStoreOp: "store",
    },
    timestampWrites: {
      querySet: queries,
      beginningOfPassWriteIndex: 2,
      endOfPassWriteIndex: 3,
    },
  });
  render.setPipeline(quad);
  render.setBindGroup(0, qb);
  render.draw(3);
  render.end();
  encoder.resolveQuerySet(queries, 0, 4, resolved, 0);
  encoder.copyBufferToBuffer(resolved, 0, readback, 0, 32);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const t = Array.from(new BigUint64Array(readback.getMappedRange()));
  readback.unmap();
  if (i >= 0) {
    computeTimes.push(Number(t[1] - t[0]) / 1e6);
    compositeTimes.push(Number(t[3] - t[2]) / 1e6);
    totalTimes.push(Number(t[3] - t[0]) / 1e6);
  }
}
function stats(a) {
  a.sort((x, y) => x - y);
  return {
    median: a[Math.floor(a.length / 2)],
    p90: a[Math.floor(a.length * 0.9)],
    min: a[0],
    max: a.at(-1),
  };
}
console.log(
  JSON.stringify(
    {
      upstream: "supersplat-viewer@34c137ab1c010b99a1c199b68c36210990331262",
      shader: "unchanged upstream WGSL",
      runtime: "Node Dawn webgpu0.6.2 Vulkan",
      adapter: {
        vendor: adapter.info.vendor,
        architecture: adapter.info.architecture,
        device: adapter.info.device,
        description: adapter.info.description,
        isFallbackAdapter: adapter.info.isFallbackAdapter,
      },
      width,
      height,
      colorFormat,
      depthFormat: "depth32float",
      iterations: computeTimes.length,
      nativeWords: words.length,
      nodes: nodeCount,
      leafWords: leafCount,
      computeGpuMs: stats(computeTimes),
      compositeGpuMs: stats(compositeTimes),
      combinedGpuMs: stats(totalTimes),
    },
    null,
    2,
  ),
);
if (cfg.image) {
  const stride = Math.ceil((width * 4) / 256) * 256,
    b = device.createBuffer({
      size: stride * height,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    }),
    e = device.createCommandEncoder();
  e.copyTextureToBuffer(
    { texture: output },
    { buffer: b, bytesPerRow: stride },
    [width, height],
  );
  device.queue.submit([e.finish()]);
  await b.mapAsync(GPUMapMode.READ);
  const all = new Uint8Array(b.getMappedRange()),
    image = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++)
    image.set(all.subarray(y * stride, y * stride + width * 4), y * width * 4);
  fs.writeFileSync(path.resolve(base, cfg.image), image);
  b.unmap();
}
device.destroy();
