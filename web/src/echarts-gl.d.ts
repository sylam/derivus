// `echarts-gl` ships no types; the two installs this client uses are `echarts.use` arguments.
declare module 'echarts-gl/charts' {
  export const SurfaceChart: (registers: unknown) => void;
}
declare module 'echarts-gl/components' {
  export const Grid3DComponent: (registers: unknown) => void;
}
