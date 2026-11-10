// Where this workstation reads the files a job names, as the Settings card edits it: the rows of
// the two maps the service keeps on this box, and the body one edit posts. Pure and free of React,
// so `scripts/files_check.mjs` drives it.

import type { PathMaps, PathsAnswer } from './types';

export type MapName = keyof PathMaps;

/** One row of a map: the key as a job writes it, and what this box reads in its place. */
export type MapRow = { written: string; here: string };

/** The key of a named file each map is looked up by. */
const KEY = { path_transform: 'folder', file_transform: 'file' } as const;

/** A map's entries, then each key a file the book names carries that the map has no entry for -
 * once, its right side empty - so a job's folder can be mapped without typing its key. */
export function mapRows(paths: PathsAnswer, map: MapName): MapRow[] {
  const entries = paths[map];
  const unmapped = paths.named.map((named) => named[KEY[map]]).filter((key) => !(key in entries));
  return [
    ...Object.entries(entries).map(([written, here]) => ({ written, here })),
    ...[...new Set(unmapped)].map((written) => ({ written, here: '' })),
  ];
}

/** Both maps whole with one entry set, an empty value removing it: `POST /paths` replaces both. */
export function edited(paths: PathMaps, map: MapName, key: string, value: string): PathMaps {
  const entries = { ...paths[map], [key]: value };
  if (value === '') delete entries[key];
  return {
    path_transform: paths.path_transform, file_transform: paths.file_transform, [map]: entries,
  };
}
