import { useEffect, useRef } from 'react';
import type { EChartsCoreOption, EChartsType } from 'echarts/core';

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

/** One hook per chart: init on mount, `setOption` on change, resize with the container,
 * dispose on unmount. No wrapper library - this is the whole of what one would do. The library is
 * fetched the first time a chart is drawn, so a tab that draws none never downloads it. */
export function useChart(option: EChartsCoreOption) {
  const container = useRef<HTMLDivElement | null>(null);
  const chart = useRef<EChartsType | null>(null);
  const latest = useRef(option);
  latest.current = option;

  useEffect(() => {
    let observer: ResizeObserver | undefined;
    let gone = false;
    import('./echarts').then(({ default: echarts }) => {
      if (gone || !container.current) return;
      chart.current = echarts.init(container.current);
      chart.current.setOption(latest.current, true);
      observer = new ResizeObserver(() => chart.current?.resize());
      observer.observe(container.current);
    });
    return () => {
      gone = true;
      observer?.disconnect();
      chart.current?.dispose();
      chart.current = null;
    };
  }, []);

  useEffect(() => {
    chart.current?.setOption(option, true);
  }, [option]);

  return container;
}
