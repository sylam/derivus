import { useEffect, useRef } from 'react';
import type { EChartsCoreOption } from 'echarts/core';
import echarts from './echarts';

/** One hook per chart: init on mount, `setOption` on change, resize with the container,
 * dispose on unmount. No wrapper library - this is the whole of what one would do. */
export function useChart(option: EChartsCoreOption) {
  const container = useRef<HTMLDivElement | null>(null);
  const chart = useRef<ReturnType<typeof echarts.init> | null>(null);

  useEffect(() => {
    if (!container.current) return;
    chart.current = echarts.init(container.current);
    const observer = new ResizeObserver(() => chart.current?.resize());
    observer.observe(container.current);
    return () => {
      observer.disconnect();
      chart.current?.dispose();
      chart.current = null;
    };
  }, []);

  useEffect(() => {
    chart.current?.setOption(option, true);
  }, [option]);

  return container;
}
