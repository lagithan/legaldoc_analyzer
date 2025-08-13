import { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { listDocuments, uploadDocument } from "@/lib/api";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { FileText, Loader2, Upload } from "lucide-react";

// Define Document type based on DocumentsListResponse
interface Document {
  id: string;
  filename: string;
  summary: string;
  risk_score: number;
  confidence_score: number;
  lawyer_recommendation: boolean;
  created_at: string;
  question_count: number;
  document_type: string;
  chromadb_stored: boolean;
  complexity_score: number;
  lawyer_urgency: string;
  legal_terms_count: number;
  risk_indicators_count: number;
  urgency_signals_count: number;
  model_used: string;
  chunk_types?: string[];
}

export default function Index() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [progress, setProgress] = useState<number>(0);
  const [progressMessage, setProgressMessage] = useState<string>("");

  useEffect(() => {
    document.title = "Dashboard • AI Legal Document Analyzer";
  }, []);

  const { data, isLoading } = useQuery({
    queryKey: ["documents"],
    queryFn: listDocuments,
  });

  // Helper function to reset upload state
  const resetUploadState = () => {
    setProgress(0);
    setProgressMessage("");
    setFile(null);
    // Reset the file input value
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    if (fileInput) {
      fileInput.value = '';
    }
  };

  const mutation = useMutation({
    mutationFn: async () => {
      if (!file) throw new Error("Please select a PDF");

      // Simulate progress stages
      setProgressMessage("Extracting Text...");
      setProgress(20);
      await new Promise((resolve) => setTimeout(resolve, 800)); // Simulate text extraction

      setProgressMessage("AI Analyzing...");
      setProgress(60);
      const result = await uploadDocument(file);

      setProgressMessage("Finalizing...");
      setProgress(90);
      await new Promise((resolve) => setTimeout(resolve, 500)); // Simulate finalization

      return result;
    },
    onSuccess: (res) => {
      setProgress(100);
      setProgressMessage("Upload Complete!");
      toast({
        title: "Success",
        description: `${res.filename} analyzed successfully.`,
        variant: "default",
        className: "bg-white text-gray-900 border-gray-300",
      });
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      navigate(`/document/${res.id}`);
      setTimeout(() => {
        resetUploadState();
      }, 1000);
    },
    onError: (err: any) => {
      // Reset progress immediately on error
      setProgress(0);
      setProgressMessage("");
      
      // Log the full error for debugging (only visible in console)
      console.error("Upload error:", err);
      
      // Extract detailed error information from server response
      let userMessage = "An unexpected error occurred. Please try again.";
      let errorTitle = "Upload Failed";
      
      try {
        // Check if error has the detailed structure from the server
        if (err.detail) {
          if (err.detail.message) {
            userMessage = err.detail.message;
          }
          if (err.detail.reason) {
            userMessage += ` ${err.detail.reason}`;
          }
          if (err.detail.suggestion) {
            userMessage += ` ${err.detail.suggestion}`;
          }
          if (err.detail.error) {
            errorTitle = err.detail.error;
          }
        }
        // Fallback to check if error message contains JSON
        else if (typeof err.message === 'string' && err.message.includes('{')) {
          try {
            const parsed = JSON.parse(err.message);
            if (parsed.detail) {
              if (parsed.detail.message) {
                userMessage = parsed.detail.message;
              }
              if (parsed.detail.reason) {
                userMessage += ` ${parsed.detail.reason}`;
              }
              if (parsed.detail.suggestion) {
                userMessage += ` ${parsed.detail.suggestion}`;
              }
              if (parsed.detail.error) {
                errorTitle = parsed.detail.error;
              }
            }
          } catch (parseError) {
            // If JSON parsing fails, fall through to other error handling
          }
        }
        // Handle other common error types
        else if (err.status_code === 422 && !err.detail) {
          userMessage = "Please upload a valid legal document.";
        } else if (err._code === 413) {
          userMessage = "File is too large. Please choose a smaller file.";
        } else if (err._code === 415) {
          userMessage = "Only PDF files are supported.";
        } else if (err._code >= 500) {
          userMessage = "Server is temporarily unavailable. Please try again later.";
        } else if (err.name === "NetworkError" || err.message?.includes("network") || err.message?.includes("fetch")) {
          userMessage = "Connection problem. Please check your internet and try again.";
        } else if (err.message?.includes("timeout")) {
          userMessage = "Upload is taking too long. Please try with a smaller file.";
        }
      } catch (processingError) {
        console.error("Error processing error message:", processingError);
        // Keep the default error message
      }
      
      toast({
        title: errorTitle,
        description: userMessage,
        variant: "destructive",
        className: "bg-white text-red-600 fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 z-50 max-w-md w-full mx-4",
        duration: 12000, // Longer duration for detailed messages
      });
      
      // Reset file state after error so user can try again
      setTimeout(() => {
        resetUploadState();
      }, 500);
    },
  });

  const recent: Document[] = useMemo(() => data?.documents?.slice(0, 5) ?? [], [data]);

  const handleClearFile = () => {
    resetUploadState();
  };

  return (
    <motion.div
      className="container mx-auto p-4 lg:p-8 space-y-6 bg-white text-gray-900"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <h1 className="text-3xl font-semibold text-gray-900 mb-6">Dashboard</h1>
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        <Card className="md:col-span-2 lg:col-span-2 bg-white border-gray-200">
          <CardHeader>
            <CardTitle className="text-gray-900">Upload a Legal PDF</CardTitle>
            <CardDescription className="text-gray-600">
              Extract text, analyze risks, and identify key clauses with AI
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Input
              type="file"
              accept="application/pdf"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              aria-label="Upload PDF"
              className="border-dashed border-gray-300 bg-gray-50 text-gray-900 focus:border-blue-500"
              disabled={mutation.isPending}
            />
            {progress > 0 && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.3 }}
                className="space-y-2"
              >
                <p className="text-sm text-gray-600">{progressMessage}</p>
                <Progress
                  value={progress}
                  className="h-2 bg-gray-200"
                />
              </motion.div>
            )}
            <div className="flex gap-3">
              <Button
                onClick={() => mutation.mutate()}
                disabled={!file || mutation.isPending}
                className="bg-black text-white"
              >
                {mutation.isPending ? (
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <Upload className="h-4 w-4 mr-2" />
                )}
                {mutation.isPending ? "Analyzing..." : "Analyze Document"}
              </Button>
              <Button
                variant="secondary"
                onClick={handleClearFile}
                disabled={!file || mutation.isPending}
                className="bg-gray-100 hover:bg-gray-200 text-gray-900 border-gray-300"
              >
                Clear
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-white border-gray-200">
          <CardHeader>
            <CardTitle className="text-gray-900">Recent Documents</CardTitle>
            <CardDescription className="text-gray-600">
              Quick access to your latest uploads
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {isLoading && (
              <p className="text-gray-600 animate-pulse flex items-center gap-2">
                <Loader2 className="h-4 w-4 animate-spin" />
                Loading...
              </p>
            )}
            {!isLoading && recent.length === 0 && (
              <div className="text-center text-gray-600">
                <FileText className="h-8 w-8 mx-auto mb-2 text-gray-400" />
                <p>No documents yet. Upload one to get started.</p>
              </div>
            )}
            {recent.map((doc, i) => (
              <motion.button
                key={doc.id}
                className="w-full text-left p-3 rounded-md border border-gray-300 hover:bg-gray-100 transition text-gray-900"
                onClick={() => navigate(`/document/${doc.id}`)}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.1 }}
              >
                <div className="font-medium truncate">{doc.filename}</div>
                <div className="text-sm text-gray-600 flex justify-between">
                  <span>Risk: {(Math.max(doc.risk_score, 0.1) * 100).toFixed(0)}%</span>
                  <span>Q&A: {doc.question_count}</span>
                </div>
                {doc.chunk_types && (
                  <div className="text-xs text-gray-500 mt-1">
                    Sections: {doc.chunk_types.join(", ")}
                  </div>
                )}
              </motion.button>
            ))}
            <Button
              variant="outline"
              className="w-full mt-2 border-gray-300 text-gray-900 hover:bg-gray-100"
              onClick={() => navigate("/analytics")}
            >
              View Analytics
            </Button>
          </CardContent>
        </Card>
      </div>
    </motion.div>
  );
}