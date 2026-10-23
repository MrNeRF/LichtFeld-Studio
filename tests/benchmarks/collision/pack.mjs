// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
export function packNative(words, rootSize) {
  const nodes = new Uint32Array(words.length + 8);
  const leaves = new Uint32Array(words.length + 2);
  let nodeCount = 1,
    leafCount = 0;
  function emit(srcIndex, n, dstIndex) {
    const word = words[srcIndex];
    if (word === 1) {
      nodes[dstIndex] = 0xff000000;
      return;
    }
    if (word === 0) throw new Error("Empty children must be omitted");
    if ((word & 0x80000000) !== 0) {
      if (n !== 4) throw new Error(`Unexpected mixed brick size ${n}`);
      const ptr = word & 0x7fffffff;
      if (leafCount / 2 >= 0x1000000)
        throw new Error("Upstream 24-bit leaf pointer overflow");
      nodes[dstIndex] = leafCount / 2;
      leaves[leafCount++] = words[ptr];
      leaves[leafCount++] = words[ptr + 1];
      return;
    }
    if (n <= 4)
      throw new Error("Native GPU data must compress mixed 4³ bricks");
    const ptr = word - 2;
    let mask = 0,
      childCount = 0;
    for (let c = 0; c < 8; c++)
      if (words[ptr + c] !== 0) {
        mask |= 1 << c;
        childCount++;
      }
    const childBase = nodeCount;
    nodeCount += childCount;
    if (childBase >= 0x1000000)
      throw new Error("Upstream 24-bit node pointer overflow");
    nodes[dstIndex] = ((mask << 24) | childBase) >>> 0;
    let offset = 0;
    for (let c = 0; c < 8; c++)
      if (mask & (1 << c)) emit(ptr + c, n / 2, childBase + offset++);
  }
  emit(0, rootSize, 0);
  return {
    packedNodes: nodes.subarray(0, nodeCount),
    packedLeaves: leaves.subarray(0, leafCount),
    nodeCount,
    leafCount,
  };
}
