import { useState } from 'react';
import { MessageSquare } from 'lucide-react';
import { DocumentUpload } from './components/DocumentUpload';
import { ChatInterface } from './components/ChatInterface';
import { resetSession } from './api/client';

export default function App() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [documentInfo, setDocumentInfo] = useState<{ name: string; size: number } | null>(null);

  const handleDocumentUpload = (file: File, nextSessionId: string) => {
    setSessionId(nextSessionId);
    setDocumentInfo({
      name: file.name,
      size: file.size,
    });
  };

  const handleRemoveDocument = async () => {
    if (sessionId) {
      try {
        await resetSession(sessionId);
      } catch {
        // Clear local UI state even if backend session cleanup fails.
      }
    }

    setSessionId(null);
    setDocumentInfo(null);
  };

  return (
    <div className="size-full flex flex-col relative overflow-hidden">
      {/* Animated background gradient */}
      <div className="absolute inset-0 bg-gradient-to-br from-emerald-50 via-green-50 to-teal-50"></div>
      
      {/* Glassy orbs for depth */}
      <div className="absolute top-20 left-10 w-96 h-96 bg-emerald-300/20 rounded-full blur-3xl"></div>
      <div className="absolute bottom-20 right-10 w-96 h-96 bg-green-300/20 rounded-full blur-3xl"></div>
      
      <div className="relative z-10 flex flex-col h-full">
        {/* Header with glassmorphism */}
        <header className="backdrop-blur-xl bg-white/30 border-b border-green-200/50 px-6 py-4 shadow-lg">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 bg-gradient-to-br from-emerald-500 to-green-600 rounded-2xl flex items-center justify-center shadow-lg backdrop-blur-sm">
              <MessageSquare className="w-7 h-7 text-white" />
            </div>
            <div>
              <h1 className="text-xl text-green-900">Document Q&A Chatbot</h1>
              <p className="text-sm text-green-700/70">Upload a document and ask questions</p>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <div className="flex-1 overflow-hidden">
          <div className="h-full max-w-7xl mx-auto p-6 flex flex-col lg:flex-row gap-6">
            {/* Left Sidebar - Document Upload */}
            <div className="lg:w-80 flex-shrink-0">
              <div className="sticky top-6">
                <h2 className="text-lg mb-4 text-green-900">Your Document</h2>
                <DocumentUpload
                  onDocumentUpload={handleDocumentUpload}
                  uploadedDocument={documentInfo}
                  onRemoveDocument={handleRemoveDocument}
                />
                
                {documentInfo && (
                  <div className="mt-4 p-4 backdrop-blur-md bg-emerald-100/60 rounded-2xl border border-emerald-300/50 shadow-lg">
                    <p className="text-sm text-emerald-900">
                      ✓ Document loaded successfully! You can now ask questions about it.
                    </p>
                  </div>
                )}
              </div>
            </div>

            {/* Right Side - Chat Interface */}
            <div className="flex-1 backdrop-blur-xl bg-white/40 rounded-3xl shadow-2xl border border-white/50 overflow-hidden flex flex-col">
              <ChatInterface
                sessionId={sessionId}
                documentName={documentInfo?.name || null}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}