import { useMemo } from 'react';
import { chartPalette } from '../charts/echarts';
import { useChart } from '../charts/useChart';
import { label } from '../tokens';
import type { TablePage } from '../types';

const MAX_SERIES = 12;

/** A date-indexed frame - an exposure profile, an mtm path - as one line per column with a zoom
 * on the date axis. Wide frames (a scenario cube) show the first MAX_SERIES columns and say so;
 * the table below the chart carries the page. */
export function TimeSeriesChart({ page }: { page: TablePage }) {
  const clipped = page.columns.length > MAX_SERIES;

  const option = useMemo(() => {
    const palette = chartPalette();
    const columns = page.columns.slice(0, MAX_SERIES);
    const dates = page.index.map((entry) => label(entry));
    return {
      animation: false,
      // the grid holds its own labels: a nine-figure exposure is wider than any fixed margin
      grid: { left: 12, right: 20, top: 30, bottom: 36, containLabel: true },
      legend: { top: 0, textStyle: { color: palette.text } },
      tooltip: { trigger: 'axis' as const },
      dataZoom: [{ type: 'inside' as const }, { type: 'slider' as const, height: 18, bottom: 8 }],
      xAxis: {
        type: 'category' as const, data: dates,
        axisLabel: { color: palette.text }, axisLine: { lineStyle: { color: palette.axis } },
      },
      yAxis: {
        type: 'value' as const, scale: true,
        axisLabel: { color: palette.text },
        splitLine: { lineStyle: { color: palette.axis, opacity: 0.5 } },
      },
      series: columns.map((column, c) => ({
        name: label(column), type: 'line' as const, symbol: 'none',
        data: page.data.map((row) => (Array.isArray(row) ? (row[c] as number) : (row as number))),
      })),
    };
  }, [page]);

  return (
    <div>
      {clipped && (
        <div className="pager">the chart draws the first {MAX_SERIES} columns</div>
      )}
      <div className="chart" ref={useChart(option)} />
    </div>
  );
}
