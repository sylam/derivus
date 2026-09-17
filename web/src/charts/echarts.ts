// The one echarts registration, tree-shaken: lines, heatmaps and the components they need. The
// 3-D surface registers `echarts-gl` onto this same core from its own lazily imported module.

import * as echarts from 'echarts/core';
// a heatmap on VALUE axes is a custom series: echarts' own needs two category axes
import { CustomChart, LineChart } from 'echarts/charts';
import {
  DataZoomComponent, GridComponent, LegendComponent, TitleComponent, TooltipComponent,
  VisualMapComponent,
} from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';

echarts.use([
  LineChart, CustomChart, GridComponent, TooltipComponent, LegendComponent,
  DataZoomComponent, VisualMapComponent, TitleComponent, CanvasRenderer,
]);

export default echarts;

/** The one continuous ramp: a heatmap's colour field, a mesh's height colouring and - sampled by
 * `ramp` - a line per expiry, so the four views of one surface read as one picture. */
export const GRADIENT = ['#3457d5', '#5b8def', '#8fd0a9', '#f2d16b', '#e2704a'];

/** The ramp at `fraction` of its length, so an ordered legend reads as the ramp itself. */
export function ramp(fraction: number): string {
  const at = Math.min(Math.max(fraction, 0), 1) * (GRADIENT.length - 1);
  const [lo, hi, weight] = [Math.floor(at), Math.ceil(at), at - Math.floor(at)];
  const channel = (i: number) => Math.round(
    parseInt(GRADIENT[lo].slice(i, i + 2), 16) * (1 - weight) +
    parseInt(GRADIENT[hi].slice(i, i + 2), 16) * weight);
  return `#${[1, 3, 5].map((i) => channel(i).toString(16).padStart(2, '0')).join('')}`;
}

/** The chart text/axis palette follows the app's own tokens, read off the live stylesheet. */
export function chartPalette() {
  const style = getComputedStyle(document.documentElement);
  return {
    text: style.getPropertyValue('--text-muted').trim() || '#5b6477',
    axis: style.getPropertyValue('--border').trim() || '#e3e7ee',
    accent: style.getPropertyValue('--accent').trim() || '#2563eb',
  };
}
