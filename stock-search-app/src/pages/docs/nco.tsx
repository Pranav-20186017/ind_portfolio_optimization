import React from 'react';
import {
  Typography,
  Container,
  Box,
  Paper,
  Grid,
  Button,
  Link as MuiLink,
  Divider
} from '@mui/material';
import Link from 'next/link';
import Head from 'next/head';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';

// Reusable Equation component for consistent math rendering
const Equation = ({ math }: { math: string }) => (
  <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 1, my: 2, textAlign: 'center' }}>
    <BlockMath math={math} />
  </Box>
);

const NestedClusteredOptimizationPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>NCO Strategy for Indian Stocks | QuantPort India Docs</title>
        <meta name="description" content="Implement Nested Clustered Optimization (NCO) for Indian stock portfolios. Combine hierarchical clustering with traditional techniques for superior NSE/BSE portfolio construction." />
        <meta property="og:title" content="NCO Strategy for Indian Stocks | QuantPort India Docs" />
        <meta property="og:description" content="Implement Nested Clustered Optimization (NCO) for Indian stock portfolios. Combine hierarchical clustering with traditional techniques for superior NSE/BSE portfolio construction." />
        <meta property="og:type" content="article" />
        <meta property="og:url" content="https://indportfoliooptimization.vercel.app/docs/nco" />
      </Head>
      
      <Container maxWidth="lg" sx={{ py: 4 }}>
        {/* Navigation Buttons */}
        <Box sx={{ mb: 4, display: 'flex', gap: 2 }}>
          <Link href="/docs" passHref>
            <Button variant="outlined" color="primary">
              ← Back to Docs
            </Button>
          </Link>
          <Link href="/" passHref>
            <Button variant="outlined" color="secondary">
              ← Back to Portfolio Optimizer
            </Button>
          </Link>
        </Box>
        
        {/* Title Section */}
        <Box sx={{ mb: 6, textAlign: 'center' }}>
          <Typography variant="h3" component="h1" gutterBottom>
            Nested Clustered Optimization (NCO)
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            A hybrid approach combining hierarchical clustering with traditional optimization
          </Typography>
        </Box>
        
        {/* Overview Section */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Overview
          </Typography>
          <Typography paragraph>
            Nested Clustered Optimization (NCO), introduced by Marcos Lopez de Prado and Michael J. Lewis in 2019, represents an innovative 
            portfolio construction framework that combines the advantages of hierarchical clustering with traditional mean-variance optimization techniques.
          </Typography>
          <Typography paragraph>
            NCO bridges the gap between purely algorithmic approaches like Hierarchical Risk Parity (HRP) and traditional optimization methods.
            It leverages the structure discovered through clustering while allowing for the incorporation of expected returns and various 
            objective functions, making it a powerful and flexible approach for modern portfolio management.
          </Typography>
        </Paper>
        
        {/* Intuitive Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Intuitive Explanation
          </Typography>
          <Typography paragraph>
            Think of NCO as a structured team decision-making process:
          </Typography>
          <Typography paragraph>
            Imagine you're managing a large corporation with many divisions. Instead of making all decisions at the top level, 
            you first organize similar divisions into departments (clustering). Then, within each department, you optimize resources 
            based on specific goals. Finally, you allocate budget across departments based on their relative importance to the company.
          </Typography>
          <Typography paragraph>
            In NCO:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                Assets are first grouped into clusters based on their similarities (like HRP).
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Within each cluster, a traditional optimization method (such as mean-variance) is applied.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                These optimized clusters are then treated as "meta-assets" for a final portfolio-wide optimization.
              </Typography>
            </li>
          </ul>
          <Typography paragraph>
            This multi-level approach combines the stability of clustering with the efficiency of optimization techniques, 
            offering a balance between structure and flexibility.
          </Typography>
          <Box sx={{ bgcolor: 'background.paper', p: 2, borderRadius: 1, border: '1px solid #e0e0e0', mt: 2 }}>
            <Typography variant="body1">
              <strong>Example:</strong> Consider a global portfolio with stocks from different countries and sectors. 
              NCO would first cluster assets by their geographic and sectoral relationships, optimize allocations within 
              each cluster (e.g., optimal allocation for European tech stocks), and then perform a final optimization 
              across these optimized clusters to create the complete portfolio.
            </Typography>
          </Box>
        </Paper>
        
        {/* Mathematical Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Detailed Mathematical & Algorithmic Explanation
          </Typography>
          <Typography paragraph>
            NCO follows a systematic approach that can be broken down into three primary stages:
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom>
            Step 1: Hierarchical Clustering
          </Typography>
          <Typography paragraph>
            Similar to HRP, NCO begins by grouping assets based on their correlations using a distance measure:
          </Typography>
          <Equation math="d_{i,j} = \sqrt{ \frac{1 - \rho_{i,j}}{2} }" />
          
          <Typography paragraph>
            where:
          </Typography>
          <ul>
            <li>
              <Typography component="div">
                <InlineMath math="d_{i,j}" /> is the distance between assets <InlineMath math="i" /> and <InlineMath math="j" />.
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <InlineMath math="\rho_{i,j}" /> is the correlation between asset returns.
              </Typography>
            </li>
          </ul>
          
          <Typography paragraph>
            This clustering creates a hierarchical structure (dendrogram) and determines which assets belong to which clusters.
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Step 2: Intra-Cluster Optimization
          </Typography>
          <Typography paragraph>
            For each identified cluster, a separate optimization problem is solved:
          </Typography>
          <Equation math="w_c^* = \arg\max_{w_c} \, f(w_c, \mu_c, \Sigma_c)" />
          
          <Typography paragraph>
            where:
          </Typography>
          <ul>
            <li>
              <Typography component="div">
                <InlineMath math="w_c^*" /> is the optimal weight vector for cluster <InlineMath math="c" />.
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <InlineMath math="f" /> is an objective function (e.g., mean-variance utility, Sharpe ratio).
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <InlineMath math="\mu_c" /> is the expected return vector for assets in cluster <InlineMath math="c" />.
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <InlineMath math="\Sigma_c" /> is the covariance matrix for assets in cluster <InlineMath math="c" />.
              </Typography>
            </li>
          </ul>
          
          <Typography paragraph>
            This step can use any of the traditional optimization approaches such as:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Maximum Sharpe ratio:</strong> Maximizing <InlineMath math="\frac{\mu^T w - r_f}{\sqrt{w^T \Sigma w}}" />
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Minimum variance:</strong> Minimizing <InlineMath math="w^T \Sigma w" />
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Mean-variance utility:</strong> Maximizing <InlineMath math="\mu^T w - \lambda w^T \Sigma w" />
              </Typography>
            </li>
          </ul>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Step 3: Inter-Cluster Allocation
          </Typography>
          <Typography paragraph>
            Each optimized cluster is now treated as a single "meta-asset" with:
          </Typography>
          <ul>
            <li>
              <Typography component="div">
                Expected return: <InlineMath math="\mu_c^* = w_c^{*T} \mu_c" />
              </Typography>
            </li>
            <li>
              <Typography component="div">
                Variance: <InlineMath math="\sigma_c^{*2} = w_c^{*T} \Sigma_c w_c^*" />
              </Typography>
            </li>
          </ul>
          
          <Typography paragraph>
            The final portfolio optimization is performed across these meta-assets:
          </Typography>
          <Equation math="v^* = \arg\max_{v} \, f(v, \mu^*, \Sigma^*)" />
          
          <Typography paragraph>
            where:
          </Typography>
          <ul>
            <li>
              <Typography component="div">
                <InlineMath math="v^*" /> is the optimal weight vector for the meta-assets.
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <InlineMath math="\mu^*" /> is the vector of expected returns for meta-assets.
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <InlineMath math="\Sigma^*" /> is the covariance matrix of meta-assets.
              </Typography>
            </li>
          </ul>
          
          <Typography paragraph>
            The final portfolio weights are computed by combining the intra-cluster weights with the inter-cluster allocation:
          </Typography>
          <Equation math="w_i = v_c^* \cdot w_{i,c}^*" />
          
          <Typography paragraph>
            where:
          </Typography>
          <ul>
            <li>
              <Typography component="div">
                <InlineMath math="w_i" /> is the final weight of asset <InlineMath math="i" /> in the portfolio.
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <InlineMath math="v_c^*" /> is the optimal weight of cluster <InlineMath math="c" /> in the portfolio.
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <InlineMath math="w_{i,c}^*" /> is the optimal weight of asset <InlineMath math="i" /> within cluster <InlineMath math="c" />.
              </Typography>
            </li>
          </ul>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Objective Functions
          </Typography>
          <Typography paragraph>
            NCO allows for various objective functions at both intra-cluster and inter-cluster levels, including:
          </Typography>
          <Box sx={{ p: 3, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', my: 3 }}>
            <ul>
              <li>
                <Typography paragraph>
                  <strong>Maximum Sharpe Ratio:</strong> Optimizing risk-adjusted return.
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <strong>Minimum Variance:</strong> Minimizing portfolio volatility.
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <strong>Mean-Variance Utility:</strong> Balancing return and risk with a risk aversion parameter.
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <strong>Maximum Diversification:</strong> Maximizing the ratio of weighted average volatility to portfolio volatility.
                </Typography>
              </li>
            </ul>
          </Box>
        </Paper>
        
        {/* Advantages and Limitations */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Advantages and Limitations
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom color="primary">
                  Advantages
                </Typography>
                <ul>
                  <li>
                    <Typography paragraph>
                      <strong>Hybrid Approach:</strong> Combines the stability of clustering with the flexibility of traditional optimization.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Incorporates Expected Returns:</strong> Unlike pure HRP, can incorporate views on asset returns.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Flexible Objective Functions:</strong> Can use different optimization criteria at different levels.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Reduced Dimensionality:</strong> By optimizing within clusters first, reduces the impact of estimation errors.
                    </Typography>
                  </li>
                </ul>
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom color="error">
                  Limitations
                </Typography>
                <ul>
                  <li>
                    <Typography paragraph>
                      <strong>Computational Complexity:</strong> More complex than both traditional optimization and pure HRP.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Parameter Sensitivity:</strong> Results depend on clustering parameters, objective functions, and expected returns.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Estimation Error Propagation:</strong> While mitigated, errors in expected returns can still impact performance.
                    </Typography>
                  </li>
                </ul>
              </Box>
            </Grid>
          </Grid>
        </Paper>
        
        {/* References */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            References
          </Typography>
          
          <ul>
            <li>
              <Typography paragraph>
                <strong>Lopez de Prado, M., & Lewis, M. J. (2019)</strong>. "Nested Clustered Optimization: A Clustering-Based Portfolio Construction Algorithm." <i>SSRN Electronic Journal</i>. doi:10.2139/ssrn.3469961
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Lopez de Prado, M. (2020)</strong>. "Machine Learning for Asset Managers," Cambridge University Press. doi:10.1017/9781108883658
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Sjöstrand, D., & Behnejad, N. (2020)</strong>. "Exploration of hierarchical clustering in long-only risk-based portfolio optimization." Master's thesis, Copenhagen Business School.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <MuiLink href="https://riskfolio-lib.readthedocs.io/en/latest/hcportfolio.html" target="_blank" rel="noopener noreferrer">
                  Riskfolio-Lib Documentation: Hierarchical Clustering Portfolio Optimization
                </MuiLink>
              </Typography>
            </li>
          </ul>
        </Paper>
        
        {/* Related Topics */}
        <Paper elevation={2} sx={{ p: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Related Topics
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Hierarchical Risk Parity (HRP)
                </Typography>
                <Typography variant="body2" paragraph>
                  The original hierarchical clustering approach for portfolio optimization.
                </Typography>
                <Link href="/docs/hrp" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Hierarchical Equal Risk Contribution (HERC)
                </Typography>
                <Typography variant="body2" paragraph>
                  Extends HRP by implementing risk parity within a hierarchical structure.
                </Typography>
                <Link href="/docs/herc" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Hierarchical Equal Risk Contribution 2 (HERC2)
                </Typography>
                <Typography variant="body2" paragraph>
                  A simplified version of HERC with equal weighting within clusters.
                </Typography>
                <Link href="/docs/herc2" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
          </Grid>
        </Paper>
      </Container>
    </>
  );
};

export default NestedClusteredOptimizationPage;

export const getStaticProps = async () => {
  return {
    props: {},
  };
}; 