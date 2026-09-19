// @ts-check
/** @type {{ id: string, title: string, load: () => Promise<any> }[]} */
export const views = [
  { id: "overview", title: "Right now", load: () => import("./overview/index.js") },
  { id: "flow", title: "Flow", load: () => import("./flow/index.js") },
  { id: "hardware", title: "Hardware", load: () => import("./hardware/index.js") },
  { id: "caching", title: "Caching", load: () => import("./caching/index.js") },
  { id: "disk", title: "Disk", load: () => import("./disk/index.js") },
  { id: "tryitout", title: "Try it out", load: () => import("./tryitout/index.js") },
];
