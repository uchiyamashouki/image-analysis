import { AnalysisWorkspace } from "./workspace.js";

const workspaceContainer = document.getElementById("workspaceContainer");
const workspaceTemplate = document.getElementById("workspaceTemplate");
const addWorkspaceBtn = document.getElementById("addWorkspaceBtn");

const workspaceList = [];

function refreshWorkspaceTitles() {
  workspaceList.forEach((ws, idx) => ws.rename(idx + 1));
}

function createWorkspace(copyFrom = null) {
  const node = workspaceTemplate.content.firstElementChild.cloneNode(true);
  workspaceContainer.appendChild(node);

  const ws = new AnalysisWorkspace(node, workspaceList.length + 1, {
    onDuplicate: duplicateWorkspace,
    onRemove: removeWorkspace
  });

  if (copyFrom) {
    ws.applyConfigFrom(copyFrom);
  }

  workspaceList.push(ws);
  refreshWorkspaceTitles();
  return ws;
}

function duplicateWorkspace(sourceWorkspace) {
  const ws = createWorkspace(sourceWorkspace);
  ws.root.scrollIntoView({ behavior: "smooth", block: "start" });
}

function removeWorkspace(workspace) {
  if (workspaceList.length === 1) {
    alert("解析ページは最低1つ残してください。");
    return;
  }

  const idx = workspaceList.indexOf(workspace);
  if (idx === -1) return;

  workspace.destroy();
  workspace.root.remove();
  workspaceList.splice(idx, 1);
  refreshWorkspaceTitles();
}

export function bootstrapApp() {
  addWorkspaceBtn.addEventListener("click", () => {
    const ws = createWorkspace();
    ws.root.scrollIntoView({ behavior: "smooth", block: "start" });
  });

  createWorkspace();
}
