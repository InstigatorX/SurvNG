export function createConditionalJsonClient(fetcher) {
  const cache = new Map();

  return {
    async get(url, errorMessage = "Unable to load data") {
      const cached = cache.get(url);
      const headers = new Headers();
      if (cached?.etag) headers.set("If-None-Match", cached.etag);
      const response = await fetcher(url, { headers });
      if (response.status === 304) {
        if (!cached) throw new Error(`${errorMessage}: cached response unavailable`);
        return cached.payload;
      }
      if (!response.ok) throw new Error(errorMessage);
      const payload = await response.json();
      const etag = response.headers.get("etag") || "";
      cache.set(url, { etag, payload });
      return payload;
    },
  };
}
