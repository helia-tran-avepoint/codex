using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AnalysisService.Models
{
    public class RoslynDeclaration
    {
        public string Name { get; set; }
        public string Type { get; set; }

        public override string ToString()
        {
            return $"Name: {Name}, Type: {Type}";
        }
    }
}
