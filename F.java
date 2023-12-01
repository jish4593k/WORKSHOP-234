import org.apache.commons.io.FileUtils;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.openai.gpt.Tokenizer;
import org.openai.gpt.TokenizerFactory;
import org.openai.gpt.models.GPT2;
import org.openai.gpt.models.GPT2Config;
import org.openai.gpt.models.GPT2Model;
import org.openai.gpt.tokenizer.SimpleTokenizer;
import org.openai.gpt.tokenizer.TextToken;
import org.openai.gpt.util.CommonPreprocessor;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

public class PdfProcessor {

    public static void main(String[] args) {
        String openaiApiKey = "<OPENAI_API_KEY>"; // Replace with your actual API key

        // Set OpenAI API key
        System.setProperty("OPENAI_API_KEY", openaiApiKey);

        // Process PDF files
        List<List<List<String>>> allChunks = processPdfFolder("./pdf", "./text");

        // Create embeddings
        OpenAIEmbeddings embeddings = new OpenAIEmbeddings();

        // Store embeddings to vector db
        FAISS db = FAISS.fromDocuments(allChunks.get(0), embeddings);
        for (List<List<String>> chunk : allChunks.subList(1, allChunks.size())) {
            FAISS dbTemp = FAISS.fromDocuments(chunk, embeddings);
            db.mergeFrom(dbTemp);
        }

        // Set up chat history and conversational retrieval chain
        List<Pair<String, String>> chatHistory = new ArrayList<>();
        ConversationalRetrievalChain qa = ConversationalRetrievalChain.fromLLM(new OpenAI(0.1), db.asRetriever());

        // Get user queries and provide responses
        while (true) {
            // Get user query
            String query = System.console().readLine("Enter a query (type 'exit' to quit): ");
            if ("exit".equalsIgnoreCase(query)) {
                break;
            }

            // Perform conversational retrieval
            INDArray result = qa.apply(Tokenizer.tokenizeSequence(query), chatHistory);

            // Update chat history
            chatHistory.add(new Pair<>(query, result.getString(0)));

            // Print the answer
            System.out.println(result.getString(0));
        }

        System.out.println("Exited!!!");
    }

    private static List<List<List<String>>> processPdfFolder(String pdfFolderPath, String txtFolderPath) {
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        // Array to hold all chunks
        List<List<List<String>>> allChunks = new ArrayList<>();

        File pdfFolder = new File(pdfFolderPath);

        // Iterate over all files in the folder
        for (File pdfFile : pdfFolder.listFiles()) {
            if (pdfFile.isFile() && pdfFile.getName().toLowerCase().endsWith(".pdf")) {
                try {
                    // Extract text from the PDF file
                    PDDocument document = PDDocument.load(pdfFile);
                    PDFTextStripper stripper = new PDFTextStripper();
                    String text = stripper.getText(document);
                    document.close();

                    // Write the extracted text to a .txt file
                    String txtFilename = pdfFile.getName().replace(".pdf", ".txt");
                    File txtFile = new File(txtFolderPath, txtFilename);
                    FileUtils.writeStringToFile(txtFile, text, StandardCharsets.UTF_8);

                    // Read the .txt file
                    String fileContent = FileUtils.readFileToString(txtFile, StandardCharsets.UTF_8);

                    // Split the text into chunks
                    List<TextToken> tokens = tokenizerFactory.create(fileContent).getTokens();
                    List<List<String>> chunks = new ArrayList<>();
                    List<String> currentChunk = new ArrayList<>();
                    int chunkSize = 512;
                    int chunkOverlap = 24;

                    for (TextToken token : tokens) {
                        currentChunk.add(token.getContent());
                        if (currentChunk.size() >= chunkSize) {
                            chunks.add(new ArrayList<>(currentChunk));
                            currentChunk.subList(0, chunkSize - chunkOverlap).clear();
                        }
                    }
                    if (!currentChunk.isEmpty()) {
                        chunks.add(currentChunk);
                    }

                    // Add chunks to the array
                    allChunks.add(chunks);

                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        return allChunks;
    }
}
