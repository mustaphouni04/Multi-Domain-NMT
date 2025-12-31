
import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";
import { FileText, Settings } from "lucide-react";
import { useTranslation } from "@/hooks/useTranslation";
import { InputSection } from "@/components/InputSection";
import { OutputSection } from "@/components/OutputSection";
import { ModelSelector } from "@/components/ModelSelector";

const Index = () => {
  const {
    sourceText,
    setSourceText,
    sourceLang,
    setSourceLang,
    translatedText,
    isTranslating,
    handleTranslate,
    handleCopyInput,
    handleCopyOutput,
    handleClear,
    handleDownload,
    swapLanguages
  } = useTranslation();

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-50 p-4">
      <div className="w-full max-w-6xl p-6 md:p-8 font-sans bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl shadow-lg">
        <header className="mb-6 text-center">
          <h1 className="text-4xl font-bold text-red-600 mb-2">
            Neural Machine Translation System
          </h1>
          <p className="text-gray-600 max-w-2xl mx-auto mb-6">
            Execute bidirectional linguistic conversion between English and Spanish utilizing our advanced
            neural network translation algorithm.
          </p>
          
          {/* Navigation */}
          <div className="flex justify-center gap-4 mb-6">
            <Button variant="default" className="bg-red-600 hover:bg-red-700">
              Text Translation
            </Button>
            <Link to="/mqxliff">
              <Button variant="outline" className="flex items-center gap-2">
                <FileText className="h-4 w-4" />
                MQXLIFF Translation
              </Button>
            </Link>
            <Link to="/training">
              <Button variant="outline" className="flex items-center gap-2">
                <Settings className="h-4 w-4" />
                Model Training
              </Button>
            </Link>
          </div>
        </header>

        {/* Model Selector */}
        <ModelSelector />
        
        <div className="grid md:grid-cols-2 gap-8">
          <InputSection
            sourceText={sourceText}
            setSourceText={setSourceText}
            sourceLang={sourceLang}
            setSourceLang={setSourceLang}
            handleCopyInput={handleCopyInput}
            handleClear={handleClear}
          />

          <OutputSection
            translatedText={translatedText}
            isTranslating={isTranslating}
            handleTranslate={handleTranslate}
            handleCopyOutput={handleCopyOutput}
            handleDownload={handleDownload}
            swapLanguages={swapLanguages}
          />
        </div>

        <footer className="mt-12 text-center text-gray-500 text-sm">
          <p>© 2023 AI Translation Engine. All rights reserved.</p>
        </footer>
      </div>
    </div>
  );
};

export default Index;
