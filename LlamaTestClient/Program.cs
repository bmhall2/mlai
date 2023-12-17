using LlamaTestClient.Prompts;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

internal class Program
{
    private async static Task Main(string[] args)
    {
        string? input;
        do
        {
            Console.WriteLine("Enter your next question: ");
            input = Console.ReadLine();
            if (input == "exit")
            {
                break;
            }

            var prompt = Prompts.LastQuestion;
            //var prompt = "You are a helpful assistant";
            
            var response = await SendTestMessageToLlama(prompt, input ?? "");
            Console.WriteLine(response);
            Console.WriteLine();

        } while (input != "exit");
    }

    private async static Task<string?> SendTestMessageToLlama(string systemMessage, string userMessage)
    {
        var http = new HttpClient()
        {
            BaseAddress = new Uri("http://localhost:5000/"),
            Timeout = TimeSpan.FromMinutes(5)
        };

        var request = new LlamaRequest(systemMessage, userMessage, 200);
        string json = JsonSerializer.Serialize(request);
        var content = new StringContent(json, Encoding.UTF8, "application/json");

        var response = await http.PostAsync("llama", content);

        string responseContent = await response.Content.ReadAsStringAsync();

        if (response.StatusCode != System.Net.HttpStatusCode.OK)
        {
            return responseContent;
        }

        var llamaResponse = JsonSerializer.Deserialize<LlamaResponse>(responseContent);

        if (llamaResponse?.Choices != null)
        {
            var rawText = llamaResponse.Choices.First().Text;
            return rawText?.Substring(rawText.IndexOf("[/INST]  ") + 9);
        }

        return null;
    }
}

public class LlamaRequest
{
    public LlamaRequest(string systemMessage, string userMessage, int maxTokens)
    {
        this.SystemMessage = systemMessage;
        this.UserMessage = userMessage;
        this.MaxTokens = maxTokens;
    }

    [JsonPropertyName("system_message")]
    public string SystemMessage { get; set; }

    [JsonPropertyName("user_message")]
    public string UserMessage { get; set; }

    [JsonPropertyName("max_tokens")]
    public int MaxTokens { get; set; }
}

public class LlamaResponse
{
    [JsonPropertyName("choices")]
    public IEnumerable<LlamaResponseChoice>? Choices { get; set; }
}

public class LlamaResponseChoice
{
    [JsonPropertyName("finish_reason")]
    public string? FinishReason { get; set; }

    [JsonPropertyName("index")]
    public int Index { get; set; }

    [JsonPropertyName("text")]
    public string? Text { get; set; }

    public string? Response { get; set; }
}