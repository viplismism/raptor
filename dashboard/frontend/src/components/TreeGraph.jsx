import { useState, useEffect, useMemo, useRef, useCallback } from "react";

const API = "/api";

export default function TreeGraph({ onClose }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [expanded, setExpanded] = useState(new Set());
  const [view, setView] = useState("explorer"); // "explorer" | "graph"

  useEffect(() => {
    fetch(`${API}/tree`)
      .then((r) => r.json())
      .then((d) => {
        if (d.error) throw new Error(d.error);
        setData(d);
        const topLayer = Math.max(...Object.keys(d.layers).map(Number));
        const rootIds = new Set(d.layers[topLayer]?.map((n) => n.index) || []);
        setExpanded(rootIds);
      })
      .catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    function onKey(e) {
      if (e.key === "Escape") {
        if (selectedNode) setSelectedNode(null);
        else onClose();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, selectedNode]);

  function toggleExpand(nodeIndex) {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(nodeIndex)) next.delete(nodeIndex);
      else next.add(nodeIndex);
      return next;
    });
  }

  function expandAll() {
    if (!data) return;
    const all = new Set();
    for (const nodes of Object.values(data.layers)) {
      for (const n of nodes) if (n.children?.length) all.add(n.index);
    }
    setExpanded(all);
  }

  function collapseAll() {
    if (!data) return;
    const topLayer = Math.max(...Object.keys(data.layers).map(Number));
    setExpanded(new Set(data.layers[topLayer]?.map((n) => n.index) || []));
  }

  const nodeMap = {};
  if (data) {
    for (const [layer, nodes] of Object.entries(data.layers)) {
      for (const n of nodes) nodeMap[n.index] = { ...n, layer: Number(layer) };
    }
  }

  const topLayer = data ? Math.max(...Object.keys(data.layers).map(Number)) : 0;
  const rootNodes = data ? (data.layers[topLayer] || []) : [];

  return (
    <div className="tree-modal-overlay" onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="tree-modal">
        <div className="tree-modal-header">
          <span className="tree-modal-title">RAPTOR Tree Structure</span>
          <div className="tree-modal-actions">
            {data && (
              <>
                <div className="tree-view-toggle">
                  <button
                    className={`tree-view-btn ${view === "explorer" ? "active" : ""}`}
                    onClick={() => setView("explorer")}
                  >
                    Explorer
                  </button>
                  <button
                    className={`tree-view-btn ${view === "graph" ? "active" : ""}`}
                    onClick={() => setView("graph")}
                  >
                    Graph
                  </button>
                </div>
                {view === "explorer" && (
                  <>
                    <button className="tree-action-btn" onClick={expandAll}>Expand all</button>
                    <button className="tree-action-btn" onClick={collapseAll}>Collapse</button>
                  </>
                )}
              </>
            )}
            <button className="tree-modal-close" onClick={onClose}>&#x2715;</button>
          </div>
        </div>
        <div className="tree-modal-body">
          {error && <div className="error">{error}</div>}
          {!data && !error && (
            <div style={{ textAlign: "center", padding: "2rem", color: "#9d9daa" }}>
              <span className="spinner" style={{ marginRight: 8 }} />
              Loading tree data...
            </div>
          )}
          {data && (
            <div className="tree-layout">
              <div className={view === "graph" ? "tree-graph-side" : "tree-explorer-side"}>
                <div className="tree-stats-bar">
                  <span className="tree-stat-pill">{Object.keys(nodeMap).length} nodes</span>
                  <span className="tree-stat-pill">{topLayer + 1} layers</span>
                  <span className="tree-stat-pill">{(data.layers[0] || []).length} chunks</span>
                </div>
                {view === "explorer" ? (
                  <div className="tree-explorer">
                    {rootNodes.map((n) => (
                      <TreeNode
                        key={n.index}
                        node={nodeMap[n.index]}
                        nodeMap={nodeMap}
                        expanded={expanded}
                        toggleExpand={toggleExpand}
                        selectedNode={selectedNode}
                        setSelectedNode={setSelectedNode}
                        topLayer={topLayer}
                        depth={0}
                      />
                    ))}
                  </div>
                ) : (
                  <GraphView
                    data={data}
                    nodeMap={nodeMap}
                    topLayer={topLayer}
                    selectedNode={selectedNode}
                    onSelectNode={setSelectedNode}
                  />
                )}
              </div>
              <div className="tree-detail-side">
                {selectedNode ? (
                  <NodeDetail node={selectedNode} topLayer={topLayer} />
                ) : (
                  <div className="node-detail-empty">Click a node to see its content</div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ── Explorer View ──────────────────────────────────────────────────────── */

function TreeNode({ node, nodeMap, expanded, toggleExpand, selectedNode, setSelectedNode, topLayer, depth }) {
  if (!node) return null;
  const hasChildren = node.children?.length > 0;
  const isExpanded = expanded.has(node.index);
  const isSelected = selectedNode?.index === node.index;
  const isLeaf = node.layer === 0;
  const isRoot = node.layer === topLayer;
  const typeClass = isLeaf ? "leaf" : isRoot ? "root" : "summary";

  return (
    <div className="tree-node-group" style={{ marginLeft: depth > 0 ? 20 : 0 }}>
      <div
        className={`tree-node-row ${isSelected ? "selected" : ""} ${typeClass}`}
        onClick={() => setSelectedNode(node)}
      >
        {hasChildren ? (
          <button className="tree-expand-btn" onClick={(e) => { e.stopPropagation(); toggleExpand(node.index); }}>
            {isExpanded ? "▾" : "▸"}
          </button>
        ) : (
          <span className="tree-expand-spacer" />
        )}
        <span className={`tree-node-dot ${typeClass}`} />
        <span className="tree-node-layer">L{node.layer}</span>
        <span className="tree-node-preview">
          {node.text.slice(0, 60).replace(/\s+/g, " ").trim()}...
        </span>
        {node.source && <span className="tree-node-source">{node.source}</span>}
      </div>
      {hasChildren && isExpanded && (
        <div className="tree-node-children">
          {node.children.map((childIdx) => (
            <TreeNode
              key={childIdx}
              node={nodeMap[childIdx]}
              nodeMap={nodeMap}
              expanded={expanded}
              toggleExpand={toggleExpand}
              selectedNode={selectedNode}
              setSelectedNode={setSelectedNode}
              topLayer={topLayer}
              depth={depth + 1}
            />
          ))}
        </div>
      )}
    </div>
  );
}

/* ── Graph View (zoomable SVG) ──────────────────────────────────────────── */

function GraphView({ data, nodeMap, topLayer, selectedNode, onSelectNode }) {
  const containerRef = useRef(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const dragStart = useRef({ x: 0, y: 0, panX: 0, panY: 0 });

  const layout = useMemo(() => {
    if (!data) return null;
    const layerKeys = Object.keys(data.layers).map(Number).sort((a, b) => a - b);

    const NODE_R = 10;
    const LAYER_HEIGHT = 120;
    const NODE_GAP = 32;
    const PADDING_X = 60;
    const PADDING_TOP = 50;
    const LABEL_WIDTH = 50;

    let maxNodesInLayer = 0;
    for (const k of layerKeys) maxNodesInLayer = Math.max(maxNodesInLayer, data.layers[k].length);

    const totalWidth = Math.max(600, maxNodesInLayer * NODE_GAP + PADDING_X * 2 + LABEL_WIDTH);
    const totalHeight = layerKeys.length * LAYER_HEIGHT + PADDING_TOP + 40;

    const positions = {};
    for (const layerNum of layerKeys) {
      const nodes = data.layers[layerNum];
      const y = totalHeight - PADDING_TOP - layerNum * LAYER_HEIGHT;
      const layerWidth = nodes.length * NODE_GAP;
      const startX = (totalWidth - LABEL_WIDTH) / 2 - layerWidth / 2 + LABEL_WIDTH;
      for (let i = 0; i < nodes.length; i++) {
        const node = nodes[i];
        positions[node.index] = {
          x: startX + i * NODE_GAP + NODE_GAP / 2,
          y,
          layer: layerNum,
          ...node,
        };
      }
    }

    const edges = [];
    for (const idx in positions) {
      const p = positions[idx];
      if (p.children) {
        for (const cIdx of p.children) {
          if (positions[cIdx]) {
            edges.push({ x1: p.x, y1: p.y, x2: positions[cIdx].x, y2: positions[cIdx].y, parentIdx: Number(idx), childIdx: cIdx });
          }
        }
      }
    }

    const labels = layerKeys.map((k) => ({
      layer: k,
      y: totalHeight - PADDING_TOP - k * LAYER_HEIGHT,
      count: data.layers[k].length,
    }));

    const maxLayer = Math.max(...layerKeys);
    return { width: totalWidth, height: totalHeight, positions, edges, labels, nodeR: NODE_R, maxLayer };
  }, [data]);

  // Zoom with scroll wheel
  const handleWheel = useCallback((e) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.1 : 0.1;
    setZoom((z) => Math.min(Math.max(0.2, z + delta), 3));
  }, []);

  useEffect(() => {
    const el = containerRef.current;
    if (el) el.addEventListener("wheel", handleWheel, { passive: false });
    return () => { if (el) el.removeEventListener("wheel", handleWheel); };
  }, [handleWheel]);

  // Pan with mouse drag
  const handleMouseDown = (e) => {
    if (e.button !== 0) return;
    setDragging(true);
    dragStart.current = { x: e.clientX, y: e.clientY, panX: pan.x, panY: pan.y };
  };
  const handleMouseMove = (e) => {
    if (!dragging) return;
    setPan({
      x: dragStart.current.panX + (e.clientX - dragStart.current.x),
      y: dragStart.current.panY + (e.clientY - dragStart.current.y),
    });
  };
  const handleMouseUp = () => setDragging(false);

  function resetView() {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }

  if (!layout) return null;

  const selIdx = selectedNode?.index;

  function nodeStyle(layer) {
    if (layer === 0) return { fill: "#eef3fc", stroke: "#3b6cce" };
    if (layer === layout.maxLayer) return { fill: "#1c1c28", stroke: "#1c1c28" };
    return { fill: "#ecf7f0", stroke: "#2d8a56" };
  }

  function isEdgeActive(e) {
    if (selIdx == null) return false;
    return e.parentIdx === selIdx || e.childIdx === selIdx;
  }

  return (
    <div className="graph-view-container">
      <div className="graph-controls">
        <button className="graph-ctrl-btn" onClick={() => setZoom((z) => Math.min(3, z + 0.2))}>+</button>
        <span className="graph-zoom-label">{Math.round(zoom * 100)}%</span>
        <button className="graph-ctrl-btn" onClick={() => setZoom((z) => Math.max(0.2, z - 0.2))}>-</button>
        <button className="graph-ctrl-btn graph-ctrl-reset" onClick={resetView}>Reset</button>
      </div>
      <div
        className="graph-canvas"
        ref={containerRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        style={{ cursor: dragging ? "grabbing" : "grab" }}
      >
        <div
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            transformOrigin: "center center",
            transition: dragging ? "none" : "transform 0.15s ease",
          }}
        >
          <svg width={layout.width} height={layout.height} viewBox={`0 0 ${layout.width} ${layout.height}`}>
            {layout.labels.map((l) => (
              <text key={l.layer} x={12} y={l.y + 4} textAnchor="start"
                style={{ fontSize: 11, fontWeight: 700, fill: "#9d9daa", textTransform: "uppercase", letterSpacing: "0.1em", fontFamily: "var(--font)" }}>
                L{l.layer} ({l.count})
              </text>
            ))}

            {layout.edges.map((e, i) => (
              <line key={i} x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2}
                stroke={isEdgeActive(e) ? "#1c1c28" : "#e8e4de"}
                strokeWidth={isEdgeActive(e) ? 2.5 : 1}
                opacity={selIdx != null && !isEdgeActive(e) ? 0.15 : 0.6}
              />
            ))}

            {Object.entries(layout.positions).map(([idx, pos]) => {
              const s = nodeStyle(pos.layer);
              const isSel = selIdx == idx;
              return (
                <circle
                  key={idx}
                  cx={pos.x}
                  cy={pos.y}
                  r={isSel ? layout.nodeR + 3 : layout.nodeR}
                  fill={s.fill}
                  stroke={s.stroke}
                  strokeWidth={isSel ? 3 : 1.5}
                  style={{ cursor: "pointer", filter: isSel ? "drop-shadow(0 0 4px rgba(0,0,0,0.3))" : "none" }}
                  onClick={(e) => { e.stopPropagation(); onSelectNode(nodeMap[idx] || pos); }}
                />
              );
            })}
          </svg>
        </div>
      </div>
    </div>
  );
}

/* ── Detail Panel ───────────────────────────────────────────────────────── */

function NodeDetail({ node, topLayer }) {
  const isLeaf = node.layer === 0;
  const isRoot = node.layer === topLayer;
  const typeClass = isLeaf ? "leaf" : isRoot ? "root" : "summary";
  const typeLabel = isLeaf ? "Leaf chunk" : isRoot ? "Root node" : "Summary";

  return (
    <div className="node-detail">
      <div className="node-detail-header">
        <span className={`node-detail-badge ${typeClass}`}>Layer {node.layer} · {typeLabel}</span>
        <span className="node-detail-id">Node {node.index}</span>
      </div>
      {node.source && (
        <div style={{ marginBottom: "0.75rem" }}>
          <span className="node-detail-source">{node.source}</span>
        </div>
      )}
      <div className="node-detail-label">Content</div>
      <div className="node-detail-text">{node.text}</div>
      {node.children?.length > 0 && (
        <>
          <div className="node-detail-label">Children ({node.children.length})</div>
          <div className="node-detail-children">
            {node.children.map((c) => <span key={c} className="node-child-tag">{c}</span>)}
          </div>
        </>
      )}
    </div>
  );
}
