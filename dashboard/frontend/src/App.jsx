import { useState, useEffect } from "react";
import Upload from "./components/Upload";
import Chat from "./components/Chat";
import TreeInfo from "./components/TreeInfo";
import TreeGraph from "./components/TreeGraph";

const API = "/api";

export default function App() {
  const [loaded, setLoaded] = useState(false);
  const [treeInfo, setTreeInfo] = useState(null);
  const [buildTimes, setBuildTimes] = useState(null);
  const [fileNames, setFileNames] = useState([]);
  const [suggestions, setSuggestions] = useState([]);
  const [showTree, setShowTree] = useState(false);
  const [uploadCount, setUploadCount] = useState(0);

  useEffect(() => {
    fetch(`${API}/status`)
      .then((r) => r.json())
      .then((data) => {
        if (data.loaded) {
          setLoaded(true);
          setTreeInfo(data.tree_info);
          setBuildTimes(data.build_times);
          setFileNames(data.file_names || []);
          setSuggestions(data.suggested_queries || []);
        }
      })
      .catch(() => {});
  }, []);

  function handleUploaded(result) {
    setLoaded(true);
    setTreeInfo(result.tree_info);
    setBuildTimes(result.build_times);
    setFileNames(result.file_names || []);
    setSuggestions(result.suggested_queries || []);
    setUploadCount((c) => c + 1);
  }

  return (
    <div className="app">
      <header className="header">
        <span className="header-logo">RAPTOR</span>
        <span className="header-sep" />
        <span className="header-sub">Hierarchical tree retrieval vs flat RAG</span>
        <div className="header-right">
          {loaded && (
            <button
              className={`header-btn ${showTree ? "active" : ""}`}
              onClick={() => setShowTree(true)}
            >
              View Tree
            </button>
          )}
        </div>
      </header>

      <main className="main">
        <Upload onUploaded={handleUploaded} loaded={loaded} fileNames={fileNames} />

        {treeInfo && <TreeInfo treeInfo={treeInfo} buildTimes={buildTimes} />}

        <Chat key={uploadCount} loaded={loaded} suggestions={suggestions} />
      </main>

      {showTree && <TreeGraph onClose={() => setShowTree(false)} />}
    </div>
  );
}
