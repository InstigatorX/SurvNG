export function browserStorage(host) {
  try {
    return host?.localStorage || null;
  } catch {
    return null;
  }
}

export function readStoredValue(storage, key, initialValue) {
  try {
    return storage?.getItem(key) || initialValue;
  } catch {
    return initialValue;
  }
}

export function writeStoredValue(storage, key, value) {
  try {
    storage?.setItem(key, value);
    return true;
  } catch {
    return false;
  }
}
