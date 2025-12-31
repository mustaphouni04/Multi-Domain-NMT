
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { RefreshCw, Copy, Download, ArrowLeftRight } from "lucide-react";

interface OutputSectionProps {
  translatedText: string;
  isTranslating: boolean;
  handleTranslate: () => void;
  handleCopyOutput: () => void;
  handleDownload: () => void;
  swapLanguages: () => void;
}

export const OutputSection = ({
  translatedText,
  isTranslating,
  handleTranslate,
  handleCopyOutput,
  handleDownload,
  swapLanguages
}: OutputSectionProps) => {
  return (
    <div className="bg-white p-6 rounded-xl shadow-md hover:shadow-xl transition-all duration-300 border border-gray-100">
      <h2 className="text-xl font-semibold mb-4">Translation</h2>
      
      <Textarea
        value={translatedText}
        readOnly
        placeholder="Translation will appear here..."
        className="min-h-[200px] mb-4 p-4 border border-gray-300 rounded-lg bg-gray-50 resize-none"
      />
      
      <div className="flex gap-4 mb-4">
        <Button
          onClick={handleTranslate}
          disabled={isTranslating}
          className="flex items-center justify-center flex-1 bg-red-600 hover:bg-red-700 text-white"
        >
          {isTranslating ? (
            <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
          ) : (
            <RefreshCw className="mr-2 h-4 w-4" />
          )}
          Translate
        </Button>
        
        <Button
          variant="outline"
          onClick={swapLanguages}
          className="flex items-center justify-center"
        >
          <ArrowLeftRight className="h-4 w-4" />
        </Button>
      </div>
      
      <div className="flex gap-4">
        <Button
          variant="outline"
          onClick={handleCopyOutput}
          disabled={!translatedText}
          className="flex items-center justify-center flex-1"
        >
          <Copy className="mr-2 h-4 w-4" />
          Copy
        </Button>
        
        <Button
          variant="outline"
          onClick={handleDownload}
          disabled={!translatedText}
          className="flex items-center justify-center flex-1"
        >
          <Download className="mr-2 h-4 w-4" />
          Download
        </Button>
      </div>
    </div>
  );
};
