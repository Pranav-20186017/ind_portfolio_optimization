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

const HierarchicalEqualRiskContributionPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Hierarchical Equal Risk Contribution (HERC) | Portfolio Optimization</title>
        <meta name="description" content="Learn about Hierarchical Equal Risk Contribution (HERC), an advanced portfolio optimization method that combines clustering with equal risk contribution principles." />
      </Head>
      
      <Container maxWidth="lg" sx={{ py: 4 }}>
        {/* Navigation Buttons */}
        <Box sx={{ mb: 4, display: 'flex', gap: 2 }}>
          <Link href="/docs" passHref>
            <Button variant="outlined" color="primary">
              ← Back to Education
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
            Hierarchical Equal Risk Contribution (HERC)
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            An advanced portfolio optimization method combining clustering with risk parity
          </Typography>
        </Box>
        
        {/* Overview Section */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Overview
          </Typography>
          <Typography paragraph>
            Hierarchical Equal Risk Contribution (HERC), introduced by Thomas Raffinot in 2017, extends the Hierarchical Risk Parity (HRP) 
            approach by incorporating principles from Equal Risk Contribution (ERC) portfolios. HERC combines the benefits of hierarchical 
            clustering for asset organization with the risk parity approach that equalizes risk contributions across portfolio components.
          </Typography>
          <Typography paragraph>
            HERC addresses some limitations of HRP by ensuring that risk is allocated more evenly across the portfolio structure, 
            potentially leading to better diversification and more robust performance across different market conditions.
          </Typography>
        </Paper>
        
        {/* Intuitive Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Intuitive Explanation
          </Typography>
          <Typography paragraph>
            Think of HERC as a balanced team organization strategy:
          </Typography>
          <Typography paragraph>
            Imagine you're organizing a large project team. First, you group team members by their complementary skills (clustering). 
            Then, instead of simply dividing work based on team size, you ensure each group bears an equal share of the project's 
            risk and complexity, regardless of how many people are in each group.
          </Typography>
          <Typography paragraph>
            In HERC:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                Assets are clustered based on their relationships (typically correlation).
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Each cluster is treated as a "risk unit" whose contribution should be equalized with other clusters.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Within clusters, risk is also distributed equally among constituent assets.
              </Typography>
            </li>
          </ul>
          <Typography paragraph>
            This multi-level risk equalization approach ensures both intra-cluster and inter-cluster risk parity, 
            creating a hierarchical structure that is both balanced and intuitively organized.
          </Typography>
          <Box sx={{ bgcolor: 'background.paper', p: 2, borderRadius: 1, border: '1px solid #e0e0e0', mt: 2 }}>
            <Typography variant="body1">
              <strong>Example:</strong> Consider a portfolio containing high-tech growth stocks, stable blue-chip stocks, bonds, and commodities. 
              Traditional methods might struggle with precise risk estimation. HERC would first group these into logical clusters, 
              then ensure each cluster contributes equally to overall portfolio risk, regardless of the number of assets in each group.
            </Typography>
          </Box>
        </Paper>
        
        {/* Mathematical Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Detailed Mathematical & Algorithmic Explanation
          </Typography>
          <Typography paragraph>
            HERC builds on the hierarchical clustering framework but implements a different allocation strategy:
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom>
            Step 1: Hierarchical Clustering
          </Typography>
          <Typography paragraph>
            Similar to HRP, HERC begins by grouping assets based on their correlations, using a distance measure:
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
            The clustering creates a hierarchical structure that groups similar assets together, forming a dendrogram.
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Step 2: Quasi-Diagonalization
          </Typography>
          <Typography paragraph>
            Assets are reordered according to the hierarchical structure to create a quasi-diagonal covariance matrix, 
            where closely related assets appear adjacent to each other.
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Step 3: Equal Risk Contribution Allocation
          </Typography>
          <Typography paragraph>
            This is where HERC diverges from HRP. Instead of recursively bisecting based on inverse variance, 
            HERC implements risk parity at each level of the hierarchy:
          </Typography>
          
          <Typography paragraph>
            <strong>Inter-Cluster Allocation:</strong> Risk is allocated equally between clusters at the same hierarchical level.
          </Typography>
          <Equation math="RC_i = RC_j \quad \forall i,j \in \text{Clusters}" />
          
          <Typography paragraph>
            where <InlineMath math="RC_i" /> is the risk contribution of cluster <InlineMath math="i" />.
          </Typography>
          
          <Typography paragraph>
            <strong>Intra-Cluster Allocation:</strong> Within each cluster, weight is determined by equalizing risk contribution:
          </Typography>
          <Equation math="w_i \cdot \frac{\partial \sigma(w)}{\partial w_i} = w_j \cdot \frac{\partial \sigma(w)}{\partial w_j} \quad \forall i,j \in \text{Cluster}" />
          
          <Typography paragraph>
            where:
          </Typography>
          <ul>
            <li>
              <Typography component="div">
                <InlineMath math="w_i" /> is the weight of asset <InlineMath math="i" />.
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <InlineMath math="\sigma(w)" /> is the portfolio volatility.
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <InlineMath math="\frac{\partial \sigma(w)}{\partial w_i}" /> is the marginal contribution to risk.
              </Typography>
            </li>
          </ul>
          
          <Typography paragraph>
            This approach ensures that each asset's contribution to its cluster's risk is equal, and each cluster's 
            contribution to the overall portfolio risk is also equal.
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Risk Measures
          </Typography>
          <Typography paragraph>
            HERC can utilize various risk measures to define risk contribution, including:
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
                  <strong>Mean Absolute Deviation (MAD):</strong> Average of absolute deviations from the mean.
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <strong>Conditional Value at Risk (CVaR):</strong> Expected loss in the worst scenarios.
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <strong>Drawdown measures:</strong> Various metrics based on portfolio drawdowns.
                </Typography>
              </li>
            </ul>
          </Box>
          
          <Typography paragraph>
            The choice of risk measure allows HERC to be adapted to different market conditions and investor preferences.
          </Typography>
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
                      <strong>Improved Risk Distribution:</strong> More consistent risk allocation across the portfolio compared to HRP.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Hierarchical Structure:</strong> Maintains the benefits of clustered asset organization from HRP.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Flexibility in Risk Measures:</strong> Can be implemented with different risk definitions beyond variance.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Robust to Estimation Errors:</strong> Less sensitive to input errors than traditional mean-variance methods.
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
                      <strong>Computational Complexity:</strong> Can be more complex to implement than HRP due to the risk parity calculations.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Parameter Sensitivity:</strong> Results can vary based on clustering method, linkage criteria, and risk measure choice.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>No Expected Return Consideration:</strong> Like HRP, focuses primarily on risk without directly incorporating return expectations.
                    </Typography>
                  </li>
                </ul>
              </Box>
            </Grid>
          </Grid>
        </Paper>
        
        {/* Comparison with Other Methods */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Comparison with Related Methods
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom>
            HERC vs. HRP
          </Typography>
          <Typography paragraph>
            <strong>Hierarchical Risk Parity (HRP)</strong> allocates based on inversely proportional risk at each split, 
            which can lead to uneven risk contributions. <strong>HERC</strong> enhances this by explicitly equalizing risk 
            contributions across clusters and within clusters, potentially providing better diversification.
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            HERC vs. Traditional Risk Parity
          </Typography>
          <Typography paragraph>
            <strong>Traditional Risk Parity</strong> treats all assets as a flat structure, equalizing risk contributions 
            across all assets. <strong>HERC</strong> respects the hierarchical structure of asset relationships, potentially 
            leading to more intuitive allocations that better reflect market segments.
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            HERC vs. Mean-Variance Optimization
          </Typography>
          <Typography paragraph>
            <strong>Mean-Variance Optimization</strong> depends heavily on expected return estimates and can produce concentrated portfolios. 
            <strong>HERC</strong> focuses on risk diversification and is more robust to estimation errors, typically producing more 
            balanced allocations.
          </Typography>
        </Paper>
        
        {/* Practical Applications */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Practical Applications and Implementation Considerations
          </Typography>
          
          <Typography paragraph>
            HERC is particularly valuable in the following scenarios:
          </Typography>
          
          <ul>
            <li>
              <Typography paragraph>
                <strong>Large Asset Universes:</strong> When dealing with many assets where estimation errors can compound.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Risk-Focused Allocation:</strong> For investors primarily concerned with risk management rather than return maximization.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Multi-Asset Portfolios:</strong> When combining different asset classes with distinct risk characteristics.
              </Typography>
            </li>
          </ul>
          
          <Typography paragraph sx={{ mt: 3 }}>
            <strong>Implementation considerations:</strong>
          </Typography>
          
          <ul>
            <li>
              <Typography paragraph>
                <strong>Linkage Method Selection:</strong> The choice of linkage method (single, complete, average, ward) can significantly 
                impact clustering results.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Risk Measure Selection:</strong> Different risk measures (variance, CVaR, etc.) can lead to different allocations.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Rebalancing Frequency:</strong> Due to the hierarchical structure, HERC portfolios may require less frequent 
                rebalancing than traditional methods.
              </Typography>
            </li>
          </ul>
        </Paper>
        
        {/* References */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            References
          </Typography>
          
          <ul>
            <li>
              <Typography paragraph>
                <strong>Raffinot, T. (2017)</strong>. "Hierarchical clustering-based asset allocation." <i>The Journal of Portfolio Management</i>, 44(2), 89-99.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Raffinot, T. (2018)</strong>. "The hierarchical equal risk contribution portfolio." <i>SSRN Electronic Journal</i>. doi:10.2139/ssrn.3237540
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Sjöstrand, D., & Behnejad, N. (2020)</strong>. "Exploration of hierarchical clustering in long-only risk-based portfolio optimization." Master's thesis, Copenhagen Business School.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>López de Prado, M. (2016)</strong>. "Building diversified portfolios that outperform out-of-sample." <i>The Journal of Portfolio Management</i>, 42(4), 59-69. doi:10.3905/jpm.2016.42.4.059
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

export default HierarchicalEqualRiskContributionPage;

export const getStaticProps = async () => {
  return {
    props: {},
  };
}; 