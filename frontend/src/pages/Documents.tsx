import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { listDocuments } from "@/lib/api";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { useToast } from "@/hooks/use-toast";
import { Loader2, FileText, AlertTriangle } from "lucide-react";

// Define the Document type to match the API response
interface Document {
  id: string;
  filename: string;
  summary: string;
  risk_score: number;
}

// Define valid Badge variants
type BadgeVariant = "default" | "secondary" | "destructive" | "outline";

// Function to determine badge variant based on risk score
const riskColor = (v: number): BadgeVariant => {
  const score = Math.max(v, 0.1); // Ensure minimum risk score of 0.1
  if (score <= 0.3) return "default";
  if (score <= 0.7) return "secondary";
  return "destructive";
};

export default function Documents() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["documents"],
    queryFn: listDocuments,
  });
  const navigate = useNavigate();
  const { toast } = useToast();

  useEffect(() => {
    document.title = "Documents • AI Legal Document Analyzer";
    console.log("Documents data:", data); // Debugging log
    if (error) {
      toast({
        title: "Error",
        description: error?.message ?? "Failed to load documents",
        variant: "destructive",
      });
    }
  }, [data, error, toast]);

  if (isLoading) {
    return (
      <div className="flex flex-col justify-center items-center h-screen">
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.5 }}
          className="text-center"
        >
          <Loader2 className="h-12 w-12 animate-spin text-blue-600 mx-auto mb-4" />
          <p className="text-slate-600 text-lg font-medium">Loading documents...</p>
        </motion.div>
      </div>
    );
  }

  if (error || !data?.documents) {
    return (
      <div className="flex flex-col justify-center items-center h-screen bg-gradient-to-br from-red-50 to-orange-50">
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.5 }}
          className="text-center"
        >
          <AlertTriangle className="h-12 w-12 text-red-500 mx-auto mb-4" />
          <p className="text-red-600 text-lg font-medium">Failed to load documents</p>
          <Button
            onClick={() => navigate("/")}
            className="mt-4"
            variant="outline"
          >
            Back to Home
          </Button>
        </motion.div>
      </div>
    );
  }

  return (
    <motion.div
      className="container py-8"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      <h1 className="text-3xl font-bold text-slate-900 mb-6">Documents</h1>
      {data.documents.length === 0 ? (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center"
        >
          <FileText className="h-12 w-12 text-slate-300 mx-auto mb-4" />
          <p className="text-slate-600 text-lg">No documents found.</p>
          <p className="text-slate-400 text-sm mt-2">Upload a document to get started.</p>
        </motion.div>
      ) : (
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {data.documents.map((d: Document, index: number) => (
            <motion.div
              key={d.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <Card className="border-0 shadow-xl bg-white/80 backdrop-blur-sm hover:shadow-md transition-shadow">
                <CardHeader>
                  <CardTitle className="text-lg truncate text-slate-900">{d.filename}</CardTitle>
                  <CardDescription className="line-clamp-2">{d.summary}</CardDescription>
                </CardHeader>
                <CardContent className="flex items-center justify-between">
                  <Badge variant={riskColor(d.risk_score)} className="text-sm font-semibold px-3 py-1">
                    Risk {(Math.max(d.risk_score, 0.1) * 100).toFixed(0)}%
                  </Badge>
                  <Button
                    variant="ghost"
                    className="text-blue-600 hover:text-blue-800 hover:underline"
                    onClick={() => navigate(`/document/${d.id}`)}
                    aria-label={`Open ${d.filename}`}
                  >
                    Open
                  </Button>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      )}
    </motion.div>
  );
}