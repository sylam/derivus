import { Suspense, lazy, useMemo, useState, type ComponentType } from 'react';
import { GRADIENT, chartPalette, ramp } from '../charts/echarts';
import { useChart } from '../charts/useChart';
import type { Descriptor } from '../types';
import {
  axisNames, mesh, neutralX, shape, sliceKeys, tenorLabel, termStructure, triples, type Shaped,
} from '../vols';

const SurfaceMesh = lazy(() => import('./SurfaceMesh'));

/** A vol as a desk reads one. */
const vol = (value: number) => `${(value * 100).toFixed(2)} %`;

type ViewProps = { shaped: Shaped; axes: string[] };

/** The shared axis furniture: quiet lines, the app's own text colour. */
function axis(palette: ReturnType<typeof chartPalette>, name: string, formatter?: unknown) {
  return {
    type: 'value' as const, name, nameTextStyle: { color: palette.text },
    scale: true, axisLine: { lineStyle: { color: palette.axis } },
    axisLabel: { color: palette.text, formatter },
    splitLine: { lineStyle: { color: palette.axis, opacity: 0.5 } },
  };
}

/** One line per expiry over the x coordinate, the legend in tenor order and coloured along the
 * ramp, so short-dated and long-dated separate at a glance. */
function Smiles({ shaped, axes }: ViewProps) {
  const option = useMemo(() => {
    const palette = chartPalette();
    const last = Math.max(shaped.smiles.length - 1, 1);
    return {
      animation: false,
      grid: { left: 62, right: 24, top: 34, bottom: 44 },
      legend: { data: shaped.smiles.map((s) => s.label), textStyle: { color: palette.text },
                top: 0, type: 'scroll' as const },
      tooltip: {
        trigger: 'axis' as const,
        valueFormatter: (value: unknown) => vol(value as number),
      },
      xAxis: axis(palette, axes[0]),
      yAxis: axis(palette, axes[2], (value: number) => vol(value)),
      series: shaped.smiles.map((smile, index) => ({
        name: smile.label, type: 'line' as const, data: smile.points,
        symbolSize: 5, smooth: 0.2,
        lineStyle: { color: ramp(index / last), width: 2 },
        itemStyle: { color: ramp(index / last) },
      })),
    };
  }, [shaped, axes]);
  return <div className="chart tall" ref={useChart(option)} />;
}

/** Vol against expiry at one x, the expiry axis labelled by tenor. */
function TermStructure({ shaped, axes }: ViewProps) {
  const [x, setX] = useState(() => neutralX(shaped.xs));
  const at = shaped.xs.includes(x) ? x : neutralX(shaped.xs);
  const option = useMemo(() => {
    const palette = chartPalette();
    return {
      animation: false,
      grid: { left: 62, right: 24, top: 20, bottom: 44 },
      tooltip: {
        trigger: 'axis' as const,
        valueFormatter: (value: unknown) => vol(value as number),
      },
      xAxis: axis(palette, axes[1], (value: number) => tenorLabel(value)),
      yAxis: axis(palette, axes[2], (value: number) => vol(value)),
      series: [{
        type: 'line' as const, data: termStructure(shaped, at), symbolSize: 6,
        lineStyle: { color: palette.accent, width: 2 }, itemStyle: { color: palette.accent },
      }],
    };
  }, [shaped, axes, at]);
  return (
    <>
      <div className="pager">
        <span>{axes[0]}</span>
        <select value={at} onChange={(event) => setX(Number(event.target.value))}>
          {shaped.xs.map((value) => <option key={value} value={value}>{value}</option>)}
        </select>
      </div>
      <div className="chart" ref={useChart(option)} />
    </>
  );
}

/** One cell's extent on a coordinate: the midpoints to its neighbours, half a gap past the ends. */
function cellSpan(values: number[], at: number): [number, number] {
  if (values.length < 2) return [at - 0.5, at + 0.5];
  const i = values.indexOf(at);
  return [i > 0 ? (values[i - 1] + at) / 2 : at - (values[1] - at) / 2,
          i < values.length - 1 ? (values[i + 1] + at) / 2
            : at + (at - values[values.length - 2]) / 2];
}

/** The heatmap the client has always drawn, over the filled mesh and on NUMERIC axes. echarts' own
 * heatmap series needs two CATEGORY axes - which is what drew this surface's clustered nodes as
 * evenly spaced strings - so a cell is a rect spanning the midpoints to its neighbours, which puts
 * every node where its coordinate says. */
function Heatmap({ shaped, axes }: ViewProps) {
  const option = useMemo(() => {
    const palette = chartPalette();
    const grid = mesh(shaped);
    const values = grid.z.flat();
    return {
      animation: false,
      grid: { left: 62, right: 96, top: 20, bottom: 44 },
      tooltip: {
        formatter: (point: { value: number[] }) =>
          `${axes[1]} ${tenorLabel(point.value[1])}<br/>${axes[0]} ${point.value[0]}` +
          `<br/>${axes[2]} ${vol(point.value[2])}`,
      },
      // the axis ends at the outermost CELL edge, so no half cell is clipped away
      xAxis: { ...axis(palette, axes[0]),
               min: cellSpan(grid.xs, grid.xs[0])[0],
               max: cellSpan(grid.xs, grid.xs[grid.xs.length - 1])[1] },
      yAxis: { ...axis(palette, axes[1]),
               min: cellSpan(grid.expiries, grid.expiries[0])[0],
               max: cellSpan(grid.expiries, grid.expiries[grid.expiries.length - 1])[1],
               axisLabel: { color: palette.text, formatter: (v: number) => tenorLabel(v) } },
      visualMap: {
        min: Math.min(...values), max: Math.max(...values), dimension: 2,
        calculable: true, orient: 'vertical' as const, right: 0, top: 'center',
        textStyle: { color: palette.text }, formatter: (value: number) => vol(value),
        inRange: { color: GRADIENT },
      },
      series: [{
        type: 'custom' as const,
        renderItem: (_: unknown, api: {
          value(index: number): number;
          coord(point: number[]): number[];
          visual(key: string): string;
        }) => {
          const [x, expiry] = [api.value(0), api.value(1)];
          const [left, right] = cellSpan(grid.xs, x);
          const [low, high] = cellSpan(grid.expiries, expiry);
          const [x0, y0] = api.coord([left, low]);
          const [x1, y1] = api.coord([right, high]);
          return {
            type: 'rect',
            shape: { x: Math.min(x0, x1), y: Math.min(y0, y1),
                     width: Math.abs(x1 - x0), height: Math.abs(y1 - y0) },
            style: { fill: api.visual('color') },
          };
        },
        data: grid.expiries.flatMap((expiry, row) =>
          grid.xs.map((x, column) => [x, expiry, grid.z[row][column]])),
      }],
    };
  }, [shaped, axes]);
  return <div className="chart tall" ref={useChart(option)} />;
}

/** The mesh, behind its own chunk: the GL code is fetched the first time a desk asks for it. */
function Mesh({ shaped, axes }: ViewProps) {
  const grid = useMemo(() => mesh(shaped), [shaped]);
  return (
    <Suspense fallback={<div className="chart tall placeholder">loading the mesh…</div>}>
      <SurfaceMesh mesh={grid} axes={axes} />
    </Suspense>
  );
}

const VIEWS: { id: string; label: string; view: ComponentType<ViewProps> }[] = [
  { id: 'smiles', label: 'smiles', view: Smiles },
  { id: 'term', label: 'term structure', view: TermStructure },
  { id: 'mesh', label: 'surface', view: Mesh },
  { id: 'heatmap', label: 'heatmap', view: Heatmap },
];

/** A shaped `.Curve` of arity 3 (`[x, expiry, value]`) or 4 (`[x, expiry, tenor, value]`) as the
 * four views of one surface, over `vols.ts`. The axis NAMES are the descriptor's own opening
 * tuple, so a `(delivery, expiry, moneyness, volatility)` grid labels itself; an arity-4 surface
 * picks one slice with a select. */
export function VolSurface({ data, descriptor }: { data: number[][]; descriptor?: Descriptor }) {
  const arity = data[0]?.length ?? 0;
  const axes = axisNames(descriptor?.description, arity);
  const keys = useMemo(() => sliceKeys(data), [data]);
  const [key, setKey] = useState<number | null>(null);
  const [tab, setTab] = useState(VIEWS[0].id);
  const slice = keys.length ? (keys.includes(key as number) ? key! : keys[0]) : undefined;
  const shaped = useMemo(() => shape(triples(data, slice)), [data, slice]);
  const Active = VIEWS.find((view) => view.id === tab)?.view ?? Smiles;
  if (!shaped.expiries.length) return null;

  return (
    <div>
      <div className="pager">
        {VIEWS.map((view) => (
          <button key={view.id} className={`chip${tab === view.id ? ' on' : ''}`}
                  onClick={() => setTab(view.id)}>{view.label}</button>
        ))}
        {keys.length > 0 && (
          <>
            <span>{axes[2]}</span>
            <select value={slice} onChange={(event) => setKey(Number(event.target.value))}>
              {keys.map((value) => <option key={value} value={value}>{value}</option>)}
            </select>
          </>
        )}
        <span>{shaped.expiries.length} expiries · {data.length} nodes</span>
      </div>
      <Active shaped={shaped} axes={arity === 4 ? [axes[0], axes[1], axes[3]] : axes} />
    </div>
  );
}
