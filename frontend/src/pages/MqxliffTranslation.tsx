
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";
import { ArrowLeft, Settings, Upload, Download, RefreshCw, Copy } from "lucide-react";
import { toast } from "@/hooks/use-toast";
import { translateMqxliff } from "@/services/translationService";
import { ModelSelector } from "@/components/ModelSelector";

const MqxliffTranslation = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [sourceLang, setSourceLang] = useState("english");
  const [isTranslating, setIsTranslating] = useState(false);
  const [translatedContent, setTranslatedContent] = useState("");

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      toast({
        description: `File "${file.name}" uploaded successfully!`,
      });
    }
  };

  const handleTranslate = async () => {
    if (!selectedFile) {
      toast({
        description: "Please select a file to translate.",
        variant: "destructive",
      });
      return;
    }

    setIsTranslating(true);
    
    try {
      const fileContent = await selectedFile.text();
      
      const response = await translateMqxliff({
        fileContent,
        fileName: selectedFile.name,
        sourceLang,
        targetLang: sourceLang === 'english' ? 'spanish' : 'english'
      });

      if (response.success) {
        setTranslatedContent(response.translatedContent || "");
        toast({
          description: "MQXLIFF translation completed successfully!",
        });
      } else {
        throw new Error(response.error || 'Translation failed');
      }
    } catch (error) {
      console.error('MQXLIFF translation error:', error);
      toast({
        description: error instanceof Error ? error.message : "Translation failed. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsTranslating(false);
    }
  };

  const handleDownload = () => {
    if (!translatedContent) {
      toast({
        description: "No translated content to download",
        variant: "destructive",
      });
      return;
    }

    const blob = new Blob([translatedContent], { type: "application/xml" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `translated_${selectedFile?.name || 'file.mqxliff'}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    toast({
      description: "Translated file downloaded!",
    });
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(translatedContent);
    toast({
      description: "Translated content copied to clipboard!",
    });
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-50 p-4">
      <div className="w-full max-w-6xl p-6 md:p-8 font-sans bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl shadow-lg">
        <header className="mb-6 text-center">
          <h1 className="text-4xl font-bold text-red-600 mb-2">
            MQXLIFF Translation
          </h1>
          <p className="text-gray-600 max-w-2xl mx-auto mb-6">
            Translate MQXLIFF files using our neural machine translation system.
          </p>
          
          {/* Navigation */}
          <div className="flex justify-center gap-4 mb-6">
            <Link to="/">
              <Button variant="outline" className="flex items-center gap-2">
                <ArrowLeft className="h-4 w-4" />
                Text Translation
              </Button>
            </Link>
            <Button variant="default" className="bg-red-600 hover:bg-red-700">
              MQXLIFF Translation
            </Button>
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
          {/* File Upload Section */}
          <div className="bg-white p-6 rounded-xl shadow-md hover:shadow-xl transition-all duration-300 border border-gray-100">
            <h2 className="text-xl font-semibold mb-4 flex items-center">
              <Upload className="text-indigo-500 mr-2" />
              Upload MQXLIFF File
            </h2>
            
            <div className="mb-4">
              <select 
                value={sourceLang}
                onChange={(e) => setSourceLang(e.target.value)}
                className="w-full py-3 px-4 appearance-none bg-white border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-400 focus:border-indigo-400 outline-none transition-all duration-200 font-medium text-center"
              >
                <option value="english">English to Spanish</option>
                <option value="spanish">Spanish to English</option>
              </select>
            </div>

            <div className="mb-4">
              <input
                type="file"
                accept=".mqxliff,.xliff"
                onChange={handleFileUpload}
                className="w-full p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-400 focus:border-indigo-400 outline-none"
              />
            </div>

            {selectedFile && (
              <div className="mb-4 p-4 bg-gray-50 rounded-lg">
                <p className="text-sm text-gray-600">Selected file:</p>
                <p className="font-medium">{selectedFile.name}</p>
                <p className="text-sm text-gray-500">{(selectedFile.size / 1024).toFixed(2)} KB</p>
              </div>
            )}

            <Button
              onClick={handleTranslate}
              disabled={isTranslating || !selectedFile}
              className="w-full flex items-center justify-center bg-red-600 hover:bg-red-700 text-white"
            >
              {isTranslating ? (
                <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <RefreshCw className="mr-2 h-4 w-4" />
              )}
              Translate MQXLIFF
            </Button>
          </div>

          {/* Translation Output Section */}
          <div className="bg-white p-6 rounded-xl shadow-md hover:shadow-xl transition-all duration-300 border border-gray-100">
            <h2 className="text-xl font-semibold mb-4 flex items-center">
              <Download className="text-indigo-500 mr-2" />
              Translation Result
            </h2>
            
            <div className="mb-4">
              <div className="w-full h-64 p-4 border border-gray-300 rounded-lg bg-gray-50 overflow-auto">
                {translatedContent ? (
                  <pre className="text-sm whitespace-pre-wrap">{translatedContent.substring(0, 500)}...</pre>
                ) : (
                  <p className="text-gray-500 italic">Translated MQXLIFF content will appear here...</p>
                )}
              </div>
            </div>
            
            <div className="flex gap-4">
              <Button
                variant="outline"
                onClick={handleCopy}
                disabled={!translatedContent}
                className="flex items-center justify-center flex-1"
              >
                <Copy className="mr-2 h-4 w-4" />
                Copy
              </Button>
              
              <Button
                variant="outline"
                onClick={handleDownload}
                disabled={!translatedContent}
                className="flex items-center justify-center flex-1"
              >
                <Download className="mr-2 h-4 w-4" />
                Download
              </Button>
            </div>
          </div>
        </div>

        <footer className="mt-12 text-center text-gray-500 text-sm">
          <p>© 2023 AI Translation Engine. All rights reserved.</p>
        </footer>
      </div>
    </div>
  );
};

export default MqxliffTranslation;
