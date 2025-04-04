using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace AnalysisService.RoslynSyntax
{
    public class VariableMerger : CSharpSyntaxRewriter
    {
        public override SyntaxNode VisitLocalDeclarationStatement(LocalDeclarationStatementSyntax node)
        {
            var variables = node.Declaration.Variables;
            if (variables.Count > 1)
                return node;

            var siblingStatements = node.Parent.ChildNodes().OfType<LocalDeclarationStatementSyntax>().ToList();
            var sameTypeStatements = siblingStatements
                .Where(stmt => stmt.Declaration.Type.ToString() == node.Declaration.Type.ToString())
                .ToList();

            if (sameTypeStatements.Count > 1)
            {
                var combinedVariables = sameTypeStatements.SelectMany(stmt => stmt.Declaration.Variables).ToList();
                var mergedDeclaration = SyntaxFactory.LocalDeclarationStatement(
                    SyntaxFactory.VariableDeclaration(node.Declaration.Type, SyntaxFactory.SeparatedList(combinedVariables))
                );

                var newRoot = node.SyntaxTree.GetRoot().ReplaceNodes(sameTypeStatements, (n1, n2) => mergedDeclaration);
                return newRoot;
            }

            return base.VisitLocalDeclarationStatement(node);
        }
    }

}
