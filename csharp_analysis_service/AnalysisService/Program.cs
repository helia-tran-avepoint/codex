using System.Net;

var builder = WebApplication.CreateBuilder(args);

var port = "8003";
if (File.Exists("./shared/service_port_mapping"))
{
    var port_mapping = File.ReadAllLines("./shared/service_port_mapping");
    port = port_mapping.FirstOrDefault(m => m.StartsWith("analysis_service"))?.Split(":").ElementAtOrDefault(1) ?? "8003";
}

// Add services to the container.

builder.Services.AddControllers();
// Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

builder.WebHost.ConfigureKestrel(serverOptions =>
{
    serverOptions.Listen(IPAddress.Parse("0.0.0.0"), int.Parse(port));
});

var app = builder.Build();

app.UseAuthorization();

app.MapControllers();

app.Run();
