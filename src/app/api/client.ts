const viteEnv = (import.meta as { env?: { VITE_API_BASE_URL?: string } }).env;

const API_BASE_URL =
  viteEnv?.VITE_API_BASE_URL?.replace(/\/$/, '') ||
  'http://127.0.0.1:8000';

export type UploadPdfResponse = {
  session_id: string;
  chunks_indexed: number;
  message: string;
};

export type AskResponse = {
  answer: string;
  sources: string[];
};

function buildError(status: number, detail: string): Error {
  return new Error(`Request failed (${status}): ${detail}`);
}

async function extractErrorDetail(response: Response): Promise<string> {
  try {
    const data = (await response.json()) as { detail?: string };
    return data.detail || response.statusText;
  } catch {
    return response.statusText;
  }
}

export async function uploadPdf(file: File): Promise<UploadPdfResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/upload-pdf`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw buildError(response.status, await extractErrorDetail(response));
  }

  return (await response.json()) as UploadPdfResponse;
}

export async function askQuestion(sessionId: string, question: string): Promise<AskResponse> {
  const response = await fetch(`${API_BASE_URL}/ask`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      session_id: sessionId,
      question,
    }),
  });

  if (!response.ok) {
    throw buildError(response.status, await extractErrorDetail(response));
  }

  return (await response.json()) as AskResponse;
}

export async function resetSession(sessionId: string): Promise<void> {
  const url = new URL(`${API_BASE_URL}/reset-session`);
  url.searchParams.set('session_id', sessionId);

  const response = await fetch(url.toString(), {
    method: 'POST',
  });

  if (!response.ok) {
    throw buildError(response.status, await extractErrorDetail(response));
  }
}
