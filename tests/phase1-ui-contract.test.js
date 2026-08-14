import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const ROOT_URL = new URL("../", import.meta.url);

async function readProjectFile(path) {
  return readFile(new URL(path, ROOT_URL), "utf8");
}

test("the UI uses the asynchronous worker boundary instead of direct Negamax", async () => {
  const main = await readProjectFile("src/main.js");

  assert.match(main, /ClassicWorkerAgent/);
  assert.match(main, /await getAIAgent\(\)\.choose/);
  assert.match(main, /new AbortController\(\)/);
  assert.match(main, /controller\.abort\(reason\)/);
  assert.match(main, /scheduledVersion !== sessionVersion/);
  for (const command of ["undo", "restartGame", "openSetup"]) {
    assert.match(
      main,
      new RegExp(`function ${command}\\([^)]*\\) \\{[\\s\\S]{0,180}cancelAITurn`),
      `${command} must cancel active AI work`,
    );
  }
  assert.match(main, /restartButton\.disabled = !gameStarted/);
  assert.match(main, /newGameButton\.disabled = false/);
  assert.doesNotMatch(main, /import\s*\{[^}]*chooseAIMove[^}]*\}/s);
  assert.doesNotMatch(main, /from ["']\.\/ai\.js["']/);
});

test("AI versus AI has no fixed three-second delay or matching product copy", async () => {
  const [main, html, readme] = await Promise.all([
    readProjectFile("src/main.js"),
    readProjectFile("index.html"),
    readProjectFile("README.md"),
  ]);
  const productSurface = `${main}\n${html}\n${readme}`;

  assert.doesNotMatch(productSurface, /AI_VS_AI_DELAY_MS/);
  assert.doesNotMatch(productSurface, /3\s*秒(?:间隔|等待)/);
  assert.doesNotMatch(productSurface, /至少\s*3\s*秒/);
  assert.match(main, /requestAnimationFrame/);
  assert.match(html, /双方自动连续行动/);
});
