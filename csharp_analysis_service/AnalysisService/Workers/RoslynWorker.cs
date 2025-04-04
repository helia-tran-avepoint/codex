using AnalysisService.Models;

using AnalysisService.RoslynSyntax;

using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

using System.Collections.Immutable;
using System.Diagnostics.Metrics;
using System.IO;
using System.Net.Http.Json;
using System.Reflection;
using System.Reflection.Metadata;
using System.Text;
using System.Text.RegularExpressions;
using System.Xml.Linq;

namespace AnalysisService
{
    public class RoslynWorker
    {
        public async Task<List<RoslynMethodInfo>> Analysis(string solutionPath)
        {
            List<RoslynMethodInfo> methodInfos = new List<RoslynMethodInfo>();
            var solutionDir = Path.GetDirectoryName(solutionPath);
            var csFiles = Directory.EnumerateFiles(solutionDir, "*.cs", SearchOption.AllDirectories);
            // var projectFiles = Directory.EnumerateFiles(solutionDir, "*.csproj", SearchOption.AllDirectories);
            // var solutionFiles = Directory.EnumerateFiles(solutionPath, "*.sln", SearchOption.AllDirectories);

            var syntaxTrees = new List<SyntaxTree>();

            var locationMap = new Dictionary<string, SyntaxTree>();
            foreach (var file in csFiles)
            {
                var syntaxTree = CSharpSyntaxTree.ParseText(File.ReadAllText(file));
                locationMap.Add(file, syntaxTree);
                syntaxTrees.Add(syntaxTree);
            }

            var compilation = CSharpCompilation.Create("CodeAnalysis")
                .AddReferences(MetadataReference.CreateFromFile(typeof(object).Assembly.Location))
                .AddSyntaxTrees(syntaxTrees);

            int maxDegreeOfParallelism = 100;
            await Parallel.ForEachAsync(csFiles, new ParallelOptions { MaxDegreeOfParallelism = maxDegreeOfParallelism }, async (file, _) =>
            {
                var partialMethodInfos = await AnalysisFile(file, locationMap, compilation);

                methodInfos.AddRange(partialMethodInfos);
            });

            //foreach (var projectFilePath in projectFiles)
            //{
            //    var projectFile = XDocument.Load(projectFilePath);

            //    var targetFramework = projectFile.Descendants("TargetFramework").FirstOrDefault()?.Value;
            //    summaryBuilder.AppendLine($"Target Framework: {targetFramework}");

            //    var packageReferences = projectFile.Descendants("PackageReference");
            //    foreach (var package in packageReferences)
            //    {
            //        var packageName = package.Attribute("Include")?.Value;
            //        var packageVersion = package.Attribute("Version")?.Value;
            //        summaryBuilder.AppendLine($"Package: {packageName}, Version: {packageVersion}");
            //    }

            //}
            return methodInfos;
        }

        private async Task<List<RoslynMethodInfo>> AnalysisFile(string file, Dictionary<string, SyntaxTree> locationMap, CSharpCompilation compilation)
        {
            List<RoslynMethodInfo> methodInfos = new List<RoslynMethodInfo>();

            var syntaxTree = locationMap[file];
            var root = syntaxTree.GetRoot();

            //root = RemoveWhitespace(root);
            //root = MergeVariableDeclarations(root);
            //root = InlineSimpleVariables(root);

            var classDeclarations = root.DescendantNodes().OfType<ClassDeclarationSyntax>();
            foreach (var classDecl in classDeclarations)
            {
                var semanticModel = compilation.GetSemanticModel(syntaxTree);

                foreach (var methodDecl in classDecl.DescendantNodes().OfType<MethodDeclarationSyntax>())
                {
                    var symbol = semanticModel.GetDeclaredSymbol(methodDecl);
                    var returnType = symbol.ReturnType;
                    var parameters = symbol.Parameters;

                    var methodInfo = new RoslynMethodInfo()
                    {
                        FileLocation = file,
                        NamespaceName = symbol.ContainingNamespace.ToString(),
                        SourceCode = methodDecl.ToFullString(),
                        Name = symbol.Name,
                        ReturnType = returnType.ToString(),
                        Parameters = parameters.Select(p => new RoslynDeclaration() { Name = p.Name, Type = p.Type.ToString() }).ToList()
                    };

                    var localDeclarations = syntaxTree.GetRoot().DescendantNodes().OfType<VariableDeclaratorSyntax>();
                    foreach (var variable in localDeclarations)
                    {
                        var variableSymbol = semanticModel.GetDeclaredSymbol(variable);
                        if (variableSymbol is ILocalSymbol localSymbol)
                        {
                            methodInfo.Variables.Add(new RoslynDeclaration() { Name = localSymbol.Name, Type = localSymbol.Type.ToString() });
                        }
                        else if (variableSymbol is IFieldSymbol fieldSymbol)
                        {
                            methodInfo.Variables.Add(new RoslynDeclaration() { Name = fieldSymbol.Name, Type = fieldSymbol.Type.ToString() });
                        }
                        else
                        {
                            methodInfo.Variables.Add(new RoslynDeclaration() { Name = variableSymbol.Name, Type = "Unknown" });
                        }
                    }

                    var classSymbol = semanticModel.GetDeclaredSymbol(classDecl);
                    var baseType = classSymbol.BaseType;
                    if (baseType != null)
                    {
                        methodInfo.ClassName = classSymbol.Name;
                        methodInfo.ClassInheritsFrom = baseType.Name;
                    }

                    var trivia = methodDecl.GetLeadingTrivia()
                        .Where(t => t.IsKind(SyntaxKind.SingleLineDocumentationCommentTrivia) ||
                        t.IsKind(SyntaxKind.MultiLineDocumentationCommentTrivia));

                    methodInfo.Comments = string.Join("\n", trivia.Select(t => t.ToString().Trim()));

                    foreach (var invocation in methodDecl.DescendantNodes().OfType<InvocationExpressionSyntax>())
                    {
                        RoslynMethodInfo callee;

                        var calleeSymbol = semanticModel.GetSymbolInfo(invocation).Symbol as IMethodSymbol;

                        if (calleeSymbol == null)
                        {
                            callee = new RoslynMethodInfo
                            {
                                ReturnType = semanticModel.GetTypeInfo(invocation).Type?.ToString() ?? "Unknown",

                                ClassName = invocation.Expression is MemberAccessExpressionSyntax memberAccess
                                    ? semanticModel.GetTypeInfo(memberAccess.Expression).Type?.Name ?? "UnknownClass"
                                    : "UnknownClass",

                                NamespaceName = invocation.Expression is MemberAccessExpressionSyntax memberAccessExpr
                                    ? semanticModel.GetTypeInfo(memberAccessExpr.Expression).Type?.ContainingNamespace?.ToString() ?? "UnknownNamespace"
                                    : "UnknownNamespace",

                                Parameters = invocation.ArgumentList.Arguments
                                    .Select(arg =>
                                    {
                                        var typeInfo = semanticModel.GetTypeInfo(arg.Expression).Type;
                                        return new RoslynDeclaration
                                        {
                                            Name = "Unknown",
                                            Type = typeInfo?.ToString() ?? "UnknownType"
                                        };
                                    }).ToList()
                            };

                            if (invocation.Expression is MemberAccessExpressionSyntax memberAccessExpression)
                            {
                                var methodSymbol = semanticModel.GetSymbolInfo(memberAccessExpression.Name).Symbol as IMethodSymbol;

                                if (methodSymbol != null)
                                {
                                    callee = new RoslynMethodInfo
                                    {
                                        Name = methodSymbol.Name,
                                        ReturnType = methodSymbol.ReturnType.ToString(),
                                        ClassName = semanticModel.GetTypeInfo(memberAccessExpression.Expression).Type?.Name ?? "UnknownClass",
                                        NamespaceName = semanticModel.GetTypeInfo(memberAccessExpression.Expression).Type?.ContainingNamespace?.ToString() ?? "UnknownNamespace",
                                        Parameters = methodSymbol.Parameters.Select(p => new RoslynDeclaration
                                        {
                                            Name = p.Name,
                                            Type = p.Type.ToString()
                                        }).ToList()
                                    };
                                }
                                else
                                {
                                    callee = new RoslynMethodInfo
                                    {
                                        Name = memberAccessExpression.Name.ToString(),
                                        ClassName = semanticModel.GetTypeInfo(memberAccessExpression.Expression).Type?.Name ?? "UnknownClass",
                                        NamespaceName = semanticModel.GetTypeInfo(memberAccessExpression.Expression).Type?.ContainingNamespace?.ToString() ?? "UnknownNamespace",
                                        ReturnType = "Unknown",
                                        Parameters = invocation.ArgumentList.Arguments.Select(arg =>
                                        {
                                            var typeInfo = semanticModel.GetTypeInfo(arg.Expression).Type;
                                            return new RoslynDeclaration
                                            {
                                                Name = "Unknown",
                                                Type = typeInfo?.ToString() ?? "UnknownType"
                                            };
                                        }).ToList()
                                    };

                                    if (semanticModel.GetSymbolInfo(memberAccessExpression.Name).Symbol is IMethodSymbol extensionMethodSymbol && extensionMethodSymbol.IsExtensionMethod)
                                    {
                                        callee.Name = extensionMethodSymbol.Name;
                                        callee.ClassName = extensionMethodSymbol.ContainingType.Name;
                                        callee.NamespaceName = extensionMethodSymbol.ContainingNamespace.ToString();
                                        callee.ReturnType = extensionMethodSymbol.ReturnType.ToString();
                                    }
                                }
                            }

                            else if (invocation.Expression is IdentifierNameSyntax identifier)
                            {
                                var localSymbols = semanticModel.LookupSymbols(invocation.SpanStart, name: identifier.Identifier.Text);
                                var localMethod = localSymbols.OfType<IMethodSymbol>().FirstOrDefault();

                                if (localMethod != null)
                                {
                                    callee.Name = localMethod.Name;
                                    callee.ClassName = localMethod.ContainingType.Name;
                                    callee.NamespaceName = localMethod.ContainingNamespace.ToString();
                                    callee.Parameters = localMethod.Parameters.Select(p => new RoslynDeclaration
                                    {
                                        Name = p.Name,
                                        Type = p.Type.ToString()
                                    }).ToList();
                                }
                                else
                                {
                                    callee.Name = identifier.Identifier.Text;
                                }
                            }
                            else
                            {
                                callee.Name = "UnknownMethod";
                            }
                        }
                        else
                        {
                            callee = new RoslynMethodInfo
                            {
                                Name = calleeSymbol.Name,
                                ReturnType = calleeSymbol.ReturnType.ToString(),
                                ClassName = calleeSymbol.ContainingType.Name,
                                NamespaceName = calleeSymbol.ContainingNamespace.ToString(),
                                Parameters = calleeSymbol.Parameters.Select(p => new RoslynDeclaration
                                {
                                    Name = p.Name,
                                    Type = p.Type.ToString()
                                }).ToList()
                            };
                        }

                        methodInfo.Callees.Add(callee);
                    }

                    methodInfos.Add(methodInfo);
                }
            }

            if (methodInfos.Count == 0)
            {
                Console.WriteLine("Extract empty info from {0}", file);
            }

            return methodInfos;
        }

        private static SyntaxNode RemoveWhitespace(SyntaxNode root)
        {
            var triviaList = root.DescendantTrivia()
                .Where(trivia => trivia.IsKind(SyntaxKind.WhitespaceTrivia) || trivia.IsKind(SyntaxKind.EndOfLineTrivia))
                .ToList();

            return root.ReplaceTrivia(triviaList, (oldTrivia, _) => SyntaxFactory.Space);
        }

        private static SyntaxNode MergeVariableDeclarations(SyntaxNode root)
        {
            var rewriter = new VariableMerger();
            return rewriter.Visit(root);
        }

        private static SyntaxNode InlineSimpleVariables(SyntaxNode root)
        {
            var rewriter = new InlineVariableRewriter();
            return rewriter.Visit(root);
        }
    }
}
