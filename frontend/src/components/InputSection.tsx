
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Copy, Trash2 } from "lucide-react";

interface InputSectionProps {
  sourceText: string;
  setSourceText: (text: string) => void;
  sourceLang: string;
  setSourceLang: (lang: string) => void;
  handleCopyInput: () => void;
  handleClear: () => void;
}

export const InputSection = ({
  sourceText,
  setSourceText,
  sourceLang,
  setSourceLang,
  handleCopyInput,
  handleClear
}: InputSectionProps) => {
  return (
    <div className="bg-white p-6 rounded-xl shadow-md hover:shadow-xl transition-all duration-300 border border-gray-100">
      <h2 className="text-xl font-semibold mb-4">Input Text</h2>
      
      <div className="mb-4">
        <select 
          value={sourceLang}
          onChange={(e) => setSourceLang(e.target.value)}
          className="w-full py-3 px-4 appearance-none bg-white border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-400 focus:border-indigo-400 outline-none transition-all duration-200 font-medium text-center"
        >
          <option value="en">English</option>
          <option value="es">Spanish</option>
        </select>
      </div>

      <Textarea
        value={sourceText}
        onChange={(e) => setSourceText(e.target.value)}
        placeholder="Enter text to translate..."
        className="min-h-[200px] mb-4 p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-400 focus:border-indigo-400 outline-none resize-none"
      />
      
      <div className="flex gap-4">
        <Button
          variant="outline"
          onClick={handleCopyInput}
          disabled={!sourceText}
          className="flex items-center justify-center flex-1"
        >
          <Copy className="mr-2 h-4 w-4" />
          Copy
        </Button>
        
        <Button
          variant="outline"
          onClick={handleClear}
          disabled={!sourceText}
          className="flex items-center justify-center flex-1"
        >
          <Trash2 className="mr-2 h-4 w-4" />
          Clear
        </Button>
      </div>
    </div>
  );
};
