// Hashing for provenance and source identity. Uses Web Crypto (browser, worker, Node >= 20).

/** JSON with object keys sorted, so equal configs hash equally regardless of key order. */
export function stableStringify(v: unknown): string {
  if (Array.isArray(v)) return `[${v.map(stableStringify).join(',')}]`;
  if (v && typeof v === 'object') {
    const o = v as Record<string, unknown>;
    return `{${Object.keys(o)
      .filter((k) => o[k] !== undefined)
      .sort()
      .map((k) => `${JSON.stringify(k)}:${stableStringify(o[k])}`)
      .join(',')}}`;
  }
  return JSON.stringify(v);
}

export async function sha256Hex(data: Uint8Array | string): Promise<string> {
  const bytes = typeof data === 'string' ? new TextEncoder().encode(data) : data;
  const digest = await crypto.subtle.digest('SHA-256', bytes as Uint8Array<ArrayBuffer>);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

/** Short hash of a JSON-able value (16 hex chars). */
export async function hashOf(v: unknown): Promise<string> {
  return (await sha256Hex(stableStringify(v))).slice(0, 16);
}

/** Bytes hashed from each end of the file. */
export const FINGERPRINT_SPAN = 1 << 20;

/**
 * Source identity: size + SHA-256 of the first and last MiB. `read(start, end)` returns those
 * bytes (fs in Node, Blob.slice or a range request in the browser).
 */
export async function fingerprint(size: number, read: (start: number, end: number) => Promise<Uint8Array>): Promise<string> {
  const head = await read(0, Math.min(size, FINGERPRINT_SPAN));
  const tail = size > FINGERPRINT_SPAN ? await read(Math.max(FINGERPRINT_SPAN, size - FINGERPRINT_SPAN), size) : new Uint8Array();
  const both = new Uint8Array(head.length + tail.length);
  both.set(head);
  both.set(tail, head.length);
  return `v0:${size}:${(await sha256Hex(both)).slice(0, 32)}`;
}
