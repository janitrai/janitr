import { IS_BROWSER, IS_WORKER } from '../../constants.mjs';

let readFile;
let request;
let fileURLToPath;
const buffer2Uin8Array = (buf) => new Uint8Array(buf.buffer, buf.byteOffset, buf.length);
const fetchFile = async (url) => {
  if (IS_BROWSER || IS_WORKER) {
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

export { buffer2Uin8Array, fetchFile };
