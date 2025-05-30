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

const HierarchicalEqualRiskContribution2Page: React.FC = () => {
  return (
    <>
      <Head>
        <title>HERC2 for Indian Stock Portfolios | QuantPort India Docs</title>
        <meta name="description" content="Implement Hierarchical Equal Risk Contribution 2 (HERC2) for Indian equity portfolios. Optimize NSE/BSE investments using advanced clustering with simplified weighting for more stable returns." />
        <meta property="og:title" content="HERC2 for Indian Stock Portfolios | QuantPort India Docs" />
        <meta property="og:description" content="Implement Hierarchical Equal Risk Contribution 2 (HERC2) for Indian equity portfolios. Optimize NSE/BSE investments using advanced clustering with simplified weighting for more stable returns." />
        <meta property="og:type" content="article" />
        <meta property="og:url" content="https://indportfoliooptimization.vercel.app/docs/herc2" />
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
            Hierarchical Equal Risk Contribution 2 (HERC2)
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            A simplified hierarchical approach with equal weights within clusters
          </Typography>
        </Box>
        
        {/* Overview Section */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Overview
          </Typography>
          <Typography paragraph>
            Hierarchical Equal Risk Contribution 2 (HERC2) is an extension of the hierarchical clustering portfolio optimization methods 
            that combines the benefits of clustering with simplicity in weight allocation. HERC2 uses hierarchical clustering to identify 
            related assets but distributes weights equally within each cluster, rather than using complex risk parity calculations.
          </Typography>
          <Typography paragraph>
            This method provides a middle ground between the sophistication of hierarchical risk-based methods (like HRP and HERC) 
            and the simplicity of equal weighting. HERC2 preserves the natural market structure through clustering while simplifying 
            the final allocation step.
          </Typography>
        </Paper>
        
        {/* Intuitive Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Intuitive Explanation
          </Typography>
          <Typography paragraph>
            Think of HERC2 as organizing a diverse project team:
          </Typography>
          <Typography paragraph>
            Imagine you're managing a large project with many tasks. First, you identify natural groupings of related tasks (clustering). 
            But instead of complex calculations to determine how much attention each task deserves, you simply divide your attention 
            equally among tasks within each group, while still recognizing that some groups may need more overall resources than others.
          </Typography>
          <Typography paragraph>
            In HERC2:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                Assets are clustered based on their correlations or other similarity measures.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                The portfolio's capital is allocated across clusters based on their risk characteristics.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Within each cluster, capital is distributed equally among constituent assets.
              </Typography>
            </li>
          </ul>
          <Typography paragraph>
            This approach maintains the hierarchical structure's benefits while simplifying the final asset allocation step, 
            potentially reducing turnover and improving stability in the portfolio.
          </Typography>
          <Box sx={{ bgcolor: 'background.paper', p: 2, borderRadius: 1, border: '1px solid #e0e0e0', mt: 2 }}>
            <Typography variant="body1">
              <strong>Example:</strong> Consider a portfolio with international stocks clustered by region. 
              HERC2 would first determine how much to allocate to each region based on risk considerations, 
              but within each region (e.g., European stocks), all stocks would receive equal allocations of that region's budget.
            </Typography>
          </Box>
        </Paper>
        
        {/* Mathematical Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Detailed Mathematical & Algorithmic Explanation
          </Typography>
          <Typography paragraph>
            HERC2 follows a similar approach to other hierarchical methods but with a distinct allocation strategy:
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom>
            Step 1: Hierarchical Clustering
          </Typography>
          <Typography paragraph>
            Like other hierarchical methods, HERC2 begins by clustering assets based on their correlations:
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
            This step creates a hierarchical structure (dendrogram) that groups similar assets together.
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Step 2: Determining Optimal Clusters
          </Typography>
          <Typography paragraph>
            HERC2 uses various methods to determine the optimal number of clusters, including:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Gap Statistics:</strong> Comparing the within-cluster dispersion to that expected under a null reference distribution.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Silhouette Analysis:</strong> Measuring how similar an object is to its own cluster compared to other clusters.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Dendrogram Distance:</strong> Using a distance threshold to determine where to "cut" the dendrogram.
              </Typography>
            </li>
          </ul>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Step 3: Inter-Cluster Allocation
          </Typography>
          <Typography paragraph>
            HERC2 allocates capital across clusters based on their risk characteristics. This can be done using several approaches:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Equal Risk Contribution:</strong> Each cluster contributes equally to overall portfolio risk.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Inverse Variance:</strong> Allocate in proportion to the inverse of cluster variance.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Minimum Variance:</strong> Optimize to minimize the overall portfolio variance.
              </Typography>
            </li>
          </ul>
          
          <Typography paragraph>
            The allocation to each cluster <InlineMath math="c" /> is determined by the chosen risk allocation method.
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Step 4: Equal Weighting Within Clusters
          </Typography>
          <Typography paragraph>
            This is where HERC2 differs from other hierarchical methods. Within each cluster, assets receive equal weights:
          </Typography>
          <Equation math="w_{i,c} = \frac{w_c}{n_c}" />
          
          <Typography paragraph>
            where:
          </Typography>
          <ul>
            <li>
              <Typography component="div">
                <InlineMath math="w_{i,c}" /> is the weight of asset <InlineMath math="i" /> in cluster <InlineMath math="c" />.
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <InlineMath math="w_c" /> is the total weight allocated to cluster <InlineMath math="c" />.
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <InlineMath math="n_c" /> is the number of assets in cluster <InlineMath math="c" />.
              </Typography>
            </li>
          </ul>
          
          <Typography paragraph>
            This equal weighting approach within clusters simplifies the allocation process while still respecting the 
            hierarchical structure of asset relationships.
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Risk Measures
          </Typography>
          <Typography paragraph>
            Like other hierarchical methods, HERC2 can use various risk measures for the inter-cluster allocation, including:
          </Typography>
          <Box sx={{ p: 3, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', my: 3 }}>
            <ul>
              <li>
                <Typography paragraph>
                  <strong>Standard Deviation/Variance:</strong> Traditional volatility-based measures.
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <strong>Conditional Value at Risk (CVaR):</strong> Expected loss in the worst scenarios.
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <strong>Maximum Drawdown:</strong> Largest peak-to-trough decline.
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <strong>Semi-Variance:</strong> Downside risk measures that focus only on negative returns.
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
                      <strong>Simplicity:</strong> Easier to implement and explain compared to full risk parity methods.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Reduced Turnover:</strong> Equal weighting within clusters can reduce portfolio turnover.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Hierarchical Structure:</strong> Maintains the benefits of respecting market structure through clustering.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Computational Efficiency:</strong> Less computationally intensive than methods requiring optimization within clusters.
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
                      <strong>Suboptimal Intra-Cluster Allocation:</strong> Equal weighting may not be optimal for risk management within clusters.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Sensitivity to Cluster Formation:</strong> Results depend heavily on the clustering approach and parameters.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Potential Over-Allocation:</strong> May over-allocate to smaller assets within a cluster compared to risk-based approaches.
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
                <strong>Lopez de Prado, M. (2016)</strong>. "Building Diversified Portfolios that Outperform Out of Sample." <i>The Journal of Portfolio Management</i>, 42(4), 59-69. doi:10.3905/jpm.2016.42.4.059
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Raffinot, T. (2018)</strong>. "The Hierarchical Equal Risk Contribution Portfolio." <i>SSRN Electronic Journal</i>. doi:10.2139/ssrn.3237540
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
                  Nested Clustered Optimization (NCO)
                </Typography>
                <Typography variant="body2" paragraph>
                  A hybrid approach combining hierarchical clustering with traditional optimization.
                </Typography>
                <Link href="/docs/nco" passHref>
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

export default HierarchicalEqualRiskContribution2Page;

export const getStaticProps = async () => {
  return {
    props: {},
  };
}; 