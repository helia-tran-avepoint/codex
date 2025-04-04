using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AnalysisService.Models
{
    public class RoslynMethodInfo
    {
        public string Name { get; set; }
        public string ReturnType { get; set; }
        public string Description { get; set; }
        public string SourceCode { get; set; }
        public string ClassInheritsFrom { get; set; }
        public string ClassName { get; set; }
        public string NamespaceName { get; set; }
        public List<RoslynDeclaration> Parameters { get; set; }
        public List<RoslynDeclaration> Variables { get; set; }
        public string Comments { get; set; }
        public string FileLocation { get; set; }
        public List<RoslynMethodInfo> Callees { get; set; }

        public RoslynMethodInfo()
        {
            Parameters = new List<RoslynDeclaration>();
            Variables = new List<RoslynDeclaration>();
            Callees = new List<RoslynMethodInfo>();
        }

        public override string ToString()
        {
            return "### Method Info ### \n" +
                $"## Location: {FileLocation} \n" +
                $"## Namespace: {NamespaceName} \n" +
                $"## Class Name: {ClassName} \n" +
                $"## Inherits From (Base Class): {ClassInheritsFrom} \n" +
                $"## Method Name: {Name} \n" + 
                $"## Method Return Type: {ReturnType} \n" +
                $"## Parameters: {string.Join("; ", Parameters.Select(p => p.ToString()))} \n" +
                $"## Variables: {string.Join("; ", Variables.Select(v => v.ToString()))} \n" +
                $"## Callees: {string.Join("; ", Callees.Select(c => c.ToString()))} \n" +
                $"## Comments: {Comments} \n" +
                $"## Source Code: {SourceCode} \n";
        }
    }

}
