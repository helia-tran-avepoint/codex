using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AnalysisService.RoslynSyntax
{
    public class InlineVariableRewriter : CSharpSyntaxRewriter
    {
        public override SyntaxNode VisitReturnStatement(ReturnStatementSyntax node)
        {
            var parentMethod = node.FirstAncestorOrSelf<MethodDeclarationSyntax>();
            var lastAssignment = parentMethod.DescendantNodes()
                .OfType<LocalDeclarationStatementSyntax>()
                .LastOrDefault();

            if (lastAssignment != null && lastAssignment.Declaration.Variables.Count == 1)
            {
                var variableName = lastAssignment.Declaration.Variables[0].Identifier.Text;
                var inlineExpression = lastAssignment.Declaration.Variables[0].Initializer?.Value;

                if (inlineExpression != null && node.Expression.ToString() == variableName)
                    return SyntaxFactory.ReturnStatement(inlineExpression);
            }

            return base.VisitReturnStatement(node);
        }
    }

}
