import { test as base, expect } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
import { randomUUID } from "node:crypto";

// Only documents created by this test's browser are eligible for cleanup.
// Fixture teardown runs after assertion failures as well as successful tests.
export const test = base.extend({
  page: async ({ page }, use) => {
    const ids = new Set<string>();
    const pending: Promise<void>[] = [];
    const receiptFailures: string[] = [];
    const receiptDirectory = path.resolve("..", ".runtime", "browser-uploads");
    fs.mkdirSync(receiptDirectory, { recursive: true });
    const receiptPath = path.join(receiptDirectory, `${randomUUID()}.json`);
    const writeReceipts = () => {
      const temporary = receiptPath + ".tmp";
      fs.writeFileSync(
        temporary,
        JSON.stringify({
          kind: "browser-test-upload-receipts",
          uploads: [...ids].map((document_id) => ({
            document_id,
            created_by_test: true,
          })),
        }),
      );
      fs.renameSync(temporary, receiptPath);
    };
    page.on("response", (response) => {
      if (
        new URL(response.url()).pathname === "/api/documents" &&
        response.request().method() === "POST" &&
        response.status() === 202
      ) {
        pending.push(
          response
            .json()
            .then((row) => {
              if (typeof row.id === "string" && row.deduplicated === false) {
                ids.add(row.id);
                writeReceipts();
              }
            })
            .catch(() => {
              receiptFailures.push("Could not persist an upload receipt");
            }),
        );
      }
    });
    try {
      await use(page);
    } finally {
      const receipts = await Promise.allSettled(pending);
      const failures: string[] = [
        ...receiptFailures,
        ...receipts
          .filter((row) => row.status === "rejected")
          .map(() => "Could not persist an upload receipt"),
      ];
      for (const id of ids) {
        try {
          const response = await page.request.delete(`/api/documents/${id}`, {
            headers: { "X-Requested-With": "AgenticRAG" },
            timeout: 15000,
          });
          if (![200, 404].includes(response.status()))
            failures.push(
              `Could not remove this test's upload ${id}: ${response.status()}`,
            );
        } catch {
          failures.push(`Cleanup interrupted for ${id}; receipt retained`);
        }
      }
      if (failures.length) throw new Error(failures.join("; "));
    }
  },
});
export { expect };
