# Documentation

For the Python API, start with the [Python and plugin documentation index](plugins/README.md).
It links the development guide, application API, plugin/UI reference, and examples.

## Documentation website

# Documentation

The documentation is built using [Docusaurus](https://docusaurus.io/), a modern static website generator.

## Prerequisites
- [Node.js](https://nodejs.org/) (>=18.0)
- [pnpm](https://pnpm.io/installation) (Package manager)

## Local Development
All commands are run from the `docs` directory.

```bash
pnpm install
```

This command installs all dependencies.

```bash
pnpm start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

## Build

```bash
pnpm build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.
