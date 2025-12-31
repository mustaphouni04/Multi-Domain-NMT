
import { useState } from "react";
import { toast } from "@/hooks/use-toast";

export const useTranslation = () => {
  const [sourceText, setSourceText] = useState("");
  const [sourceLang, setSourceLang] = useState("en");
  const [translatedText, setTranslatedText] = useState("");
  const [isTranslating, setIsTranslating] = useState(false);

  const handleTranslate = async () => {
    if (!sourceText.trim()) {
      toast({
        description: "Please enter text to translate",
        variant: "destructive",
      });
      return;
    }

    setIsTranslating(true);
    try {
      const response = await fetch('/api/translate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: sourceText
        })
      });

      if (!response.ok) {
        throw new Error('Translation failed');
      }

      const data = await response.json();
      setTranslatedText(data.translation);
      
      toast({
        description: "Translation completed successfully",
      });
    } catch (error) {
      console.error('Translation error:', error);
      toast({
        description: "Translation failed. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsTranslating(false);
    }
  };

  const handleCopyInput = async () => {
    try {
      await navigator.clipboard.writeText(sourceText);
      toast({
        description: "Source text copied to clipboard",
      });
    } catch (error) {
      console.error('Copy failed:', error);
    }
  };

  const handleCopyOutput = async () => {
    try {
      await navigator.clipboard.writeText(translatedText);
      toast({
        description: "Translation copied to clipboard",
      });
    } catch (error) {
      console.error('Copy failed:', error);
    }
  };

  const handleClear = () => {
    setSourceText("");
    setTranslatedText("");
  };

  const handleDownload = () => {
    const element = document.createElement("a");
    const file = new Blob([translatedText], { type: 'text/plain' });
    element.href = URL.createObjectURL(file);
    element.download = "translation.txt";
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  const swapLanguages = () => {
    const newSourceLang = sourceLang === 'en' ? 'es' : 'en';
    setSourceLang(newSourceLang);
    setSourceText(translatedText);
    setTranslatedText(sourceText);
  };

  return {
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
  };
};
