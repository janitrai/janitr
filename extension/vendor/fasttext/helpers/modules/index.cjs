'use strict';

Object.defineProperty(exports, Symbol.toStringTag, { value: 'Module' });

const constants = require('../../constants.cjs');

let readFile;
let request;
let fileURLToPath;
const buffer2Uin8Array = (buf) => new Uint8Array(buf.buffer, buf.byteOffset, buf.length);
const fetchFile = async (url) => {
  if (constants.IS_BROWSER || constants.IS_WORKER) {
    return new Uint8Array(await (await fetch(url)).arrayBuffer());
  }
  if (url.startsWith("file://")) {
    readFile ??= (await import('fs/promises')).readFile;
    fileURLToPath ??= (await import('url')).fileURLToPath;
    return buffer2Uin8Array(await readFile(fileURLToPath(url)));
  } else {
    request ??= (await import('http')).request;
    return new Promise((resolve, reject) => {
      const chunks = [];
      request(
        url,
        (res) => res.on("close", () => resolve(buffer2Uin8Array(Buffer.concat(chunks)))).on("data", (chunk) => chunks.push(chunk)).on("error", (err) => reject(err))
      );
    });
  }
};

exports.buffer2Uin8Array = buffer2Uin8Array;
exports.fetchFile = fetchFile;
