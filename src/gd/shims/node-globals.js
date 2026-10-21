globalThis.global = globalThis;
globalThis.process = globalThis.process || { env: {}, argv: [], emitWarning() {}, on() {}, exit() {} };
