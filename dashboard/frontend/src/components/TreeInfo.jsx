export default function TreeInfo({ treeInfo, buildTimes }) {
  if (!treeInfo) return null;

  const stats = [
    { value: treeInfo.num_layers, label: "Layers" },
    { value: treeInfo.total_nodes, label: "Total Nodes" },
    { value: treeInfo.leaf_nodes, label: "Leaf Chunks" },
    { value: treeInfo.root_nodes, label: "Root Nodes" },
  ];

  if (buildTimes) {
    stats.push(
      { value: `${buildTimes.raptor}s`, label: "RAPTOR Build" },
      { value: `${buildTimes.flat}s`, label: "Flat Build" },
    );
  }

  return (
    <section className="tree-info">
      {stats.map((s) => (
        <div className="stat" key={s.label}>
          <div className="stat-value">{s.value}</div>
          <div className="stat-label">{s.label}</div>
        </div>
      ))}
    </section>
  );
}
