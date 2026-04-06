import { useState } from 'react';
import { Upload, FileText, X } from 'lucide-react';
import { Button } from './ui/button';
import { Card } from './ui/card';
import { uploadPdf } from '../api/client';

interface DocumentUploadProps {
  onDocumentUpload: (file: File, sessionId: string) => void;
  uploadedDocument: { name: string; size: number } | null;
  onRemoveDocument: () => Promise<void>;
}

export function DocumentUpload({ onDocumentUpload, uploadedDocument, onRemoveDocument }: DocumentUploadProps) {
  const [isUploading, setIsUploading] = useState(false);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      if (file.type !== 'application/pdf') {
        alert('Please upload a PDF file');
        return;
      }

      setIsUploading(true);
      const payload = await uploadPdf(file);
      onDocumentUpload(file, payload.session_id);
    } catch (error) {
      console.error('Error uploading file:', error);
      alert(error instanceof Error ? error.message : 'Error uploading file. Please try again.');
    } finally {
      setIsUploading(false);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  return (
    <div className="w-full">
      {!uploadedDocument ? (
        <Card className="border-dashed border-2 border-green-300/50 p-8 backdrop-blur-md bg-white/50 shadow-lg hover:shadow-xl transition-all duration-300 hover:border-green-400/60">
          <label className="flex flex-col items-center justify-center cursor-pointer">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-emerald-400 to-green-500 flex items-center justify-center mb-4 shadow-lg">
              <Upload className="w-8 h-8 text-white" />
            </div>
            <span className="text-sm text-green-800 mb-2">
              Upload a PDF to get started
            </span>
            <span className="text-xs text-green-600/70 mb-4">
              PDF files supported
            </span>
            <Button
              type="button"
              variant="outline"
              disabled={isUploading}
              className="backdrop-blur-sm bg-white/60 border-green-300/50 text-green-800 hover:bg-green-50/80"
            >
              {isUploading ? 'Uploading...' : 'Choose File'}
            </Button>
            <input
              type="file"
              className="hidden"
              accept=".pdf"
              onChange={handleFileChange}
            />
          </label>
        </Card>
      ) : (
        <Card className="p-4 backdrop-blur-md bg-white/60 border-green-300/50 shadow-lg">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-400 to-green-500 flex items-center justify-center">
                <FileText className="w-6 h-6 text-white" />
              </div>
              <div>
                <p className="text-sm text-green-900">{uploadedDocument.name}</p>
                <p className="text-xs text-green-700/70">
                  {formatFileSize(uploadedDocument.size)}
                </p>
              </div>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                void onRemoveDocument();
              }}
              className="text-green-700 hover:text-green-900 hover:bg-green-100/50"
            >
              <X className="w-4 h-4" />
            </Button>
          </div>
        </Card>
      )}
    </div>
  );
}