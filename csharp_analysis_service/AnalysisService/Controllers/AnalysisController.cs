using AnalysisService.Models;

using Microsoft.AspNetCore.Mvc;

namespace AnalysisService.Controllers
{
    [ApiController]
    public class AnalysisController : ControllerBase
    {
        private readonly RoslynWorker _roslynWorker;

        public AnalysisController()
        {
            _roslynWorker = new RoslynWorker();
        }

        [HttpPost("analysis")]
        public async Task<IActionResult> analysis([FromBody] BuildIndexRequest request)
        {
            var methodInfos = await _roslynWorker.Analysis(request.Path);
            return Ok(new { status = "success", message = "Analysis successfully", methodInfos });
        }
    }
}
