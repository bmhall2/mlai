using UglyToad.PdfPig;
using UglyToad.PdfPig.DocumentLayoutAnalysis.TextExtractor;

internal class Program
{
    private static void Main(string[] args)
    {
        var outputPath = "/mnt/c/Users/Ben Hall/code/mlai/PdfTextTest/text/";
        var files = Directory.GetFiles("/mnt/c/Users/Ben Hall/code/mlai/PdfTextTest/pdfs/");
        foreach (var file in files)
        {
            var fileName = file.Split("/").Last().Replace(".pdf", ".txt");
            using (var outputFile = File.CreateText(Path.Combine(outputPath, fileName)))
            {
                using (var pdf = PdfDocument.Open(file))
                {
                    foreach (var page in pdf.GetPages())
                    {
                        outputFile.Write(ContentOrderTextExtractor.GetText(page));
                    }
                }
            }
        }
    }
}