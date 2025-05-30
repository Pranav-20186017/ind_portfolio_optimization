import React from 'react';
import {
  Typography,
  Container,
  Box,
  Paper,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Divider,
  Button,
  Link as MuiLink
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

const MinimumVolatilityPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Minimum Volatility for Indian Stocks | QuantPort India Docs</title>
        <meta name="description" content="Build lower-risk Indian equity portfolios with Minimum Volatility optimization. Discover how to reduce NSE/BSE stock portfolio volatility while maintaining reasonable returns." />
        <meta property="og:title" content="Minimum Volatility for Indian Stocks | QuantPort India Docs" />
        <meta property="og:description" content="Build lower-risk Indian equity portfolios with Minimum Volatility optimization. Discover how to reduce NSE/BSE stock portfolio volatility while maintaining reasonable returns." />
        <meta property="og:type" content="article" />
        <meta property="og:url" content="https://indportfoliooptimization.vercel.app/docs/min-vol" />
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
            Minimum Volatility Optimization
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            Minimizing portfolio risk regardless of expected returns
          </Typography>
        </Box>
        
        {/* Overview Section */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Overview
          </Typography>
          <Typography paragraph>
            Minimum Volatility (Min Volatility) optimization is a quantitative method within Modern Portfolio Theory (MPT) 
            aimed explicitly at constructing portfolios that minimize risk (volatility) regardless of the expected returns. 
            Unlike Mean-Variance Optimization (MVO) which balances return and risk, the Min Volatility strategy 
            purely seeks the lowest-risk combination of assets.
          </Typography>
          <Typography paragraph>
            This approach represents one extreme point on the efficient frontier—the leftmost point that 
            offers the absolute minimum level of risk possible given the available assets and constraints.
          </Typography>
        </Paper>
        
        {/* Intuitive Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Intuitive Explanation
          </Typography>
          <Typography paragraph>
            Imagine building a safety-first sports team. Instead of prioritizing scoring high points (returns), 
            your primary goal is to reduce the likelihood of mistakes (volatility). This approach emphasizes 
            reliability and consistency over high performance. Similarly, Min Volatility portfolios offer 
            investors the lowest possible risk, making them ideal for conservative investors or volatile market conditions.
          </Typography>
          <Box sx={{ bgcolor: 'background.paper', p: 2, borderRadius: 1, border: '1px solid #e0e0e0', mt: 2 }}>
            <Typography variant="body1">
              <strong>Example:</strong> Consider a retirement portfolio for someone nearing retirement. 
              At this stage, preserving capital is more important than aggressive growth. A Minimum Volatility 
              approach would construct a portfolio emphasizing stability, even if it means accepting more modest returns.
            </Typography>
          </Box>
        </Paper>
        
        {/* Mathematical Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Detailed Mathematical Explanation
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom>
            Problem Setup
          </Typography>
          
          <Box sx={{ p: 3, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', my: 3 }}>
            <Typography variant="subtitle1" gutterBottom>
              <strong>Given:</strong>
            </Typography>
            <Typography component="div" paragraph>
              <InlineMath math="n" /> assets in the portfolio.
            </Typography>
            <Typography component="div" paragraph>
              Asset covariance matrix <InlineMath math="\Sigma \in \mathbb{R}^{n \times n}" />.
            </Typography>
            <Typography component="div" paragraph>
              Portfolio weights <InlineMath math="w \in \mathbb{R}^n" />, with each weight <InlineMath math="w_i" /> representing 
              the proportion of the total portfolio invested in asset <InlineMath math="i" />.
            </Typography>
          </Box>
          
          <Typography variant="h6" component="h3" gutterBottom>
            Objective Function
          </Typography>
          <Typography paragraph>
            The goal is to minimize the portfolio variance, defined as:
          </Typography>
          <Equation math="\sigma_p^2 = w^T \Sigma w = \sum_{i=1}^{n} \sum_{j=1}^{n} w_i w_j \sigma_{ij}" />
          
          <Typography paragraph>
            where:
          </Typography>
          <ul>
            <li>
              <Typography component="div">
                <InlineMath math="\sigma_p^2" /> is the portfolio variance (the squared volatility).
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <InlineMath math="w" /> is the vector of portfolio weights.
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <InlineMath math="\Sigma" /> is the covariance matrix, representing how assets move together.
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <InlineMath math="\sigma_{ij}" /> is the covariance between assets <InlineMath math="i" /> and <InlineMath math="j" />.
              </Typography>
            </li>
          </ul>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Constraints
          </Typography>
          
          <Typography variant="subtitle1" gutterBottom>
            <strong>Full Investment Constraint</strong>
          </Typography>
          <Typography paragraph>
            The sum of the portfolio weights must be exactly 1 (fully invested portfolio):
          </Typography>
          <Equation math="\sum_{i=1}^{n} w_i = 1" />
          
          <Typography variant="subtitle1" gutterBottom>
            <strong>No Short Selling (Optional)</strong>
          </Typography>
          <Typography paragraph>
            Each asset weight is non-negative (no negative investment):
          </Typography>
          <Equation math="w_i \geq 0 \quad \forall i" />
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Complete Mathematical Formulation
          </Typography>
          <Typography paragraph>
            Thus, the mathematical formulation for the Minimum Volatility optimization is:
          </Typography>
          <Equation math="\begin{aligned} \min_{w} \quad & w^T \Sigma w \\ \text{subject to} \quad & \mathbf{1}^T w = 1 \\ & w_i \geq 0, \quad i = 1, \dots, n \end{aligned}" />
          
          <Typography paragraph>
            This is a quadratic programming problem with linear constraints that can be solved efficiently 
            using specialized optimization algorithms like interior-point methods or active-set methods.
          </Typography>
        </Paper>
        
        {/* Key Differences from MVO */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Key Differences from Mean-Variance Optimization
          </Typography>
          <Typography paragraph>
            Unlike Mean-Variance Optimization (MVO), the Minimum Volatility approach:
          </Typography>
          
          <Box sx={{ mt: 2 }}>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Paper elevation={0} sx={{ bgcolor: '#f8f9fa', p: 2, height: '100%' }}>
                  <Typography variant="h6" gutterBottom>
                    Mean-Variance Optimization
                  </Typography>
                  <ul>
                    <li><Typography paragraph>Balances risk and return</Typography></li>
                    <li><Typography paragraph>Requires expected returns estimates</Typography></li>
                    <li><Typography paragraph>Solutions are sensitive to expected return inputs</Typography></li>
                    <li><Typography paragraph>Offers a range of efficient portfolios</Typography></li>
                    <li><Typography paragraph>Focuses on maximizing the Sharpe ratio or optimizing for target returns</Typography></li>
                  </ul>
                </Paper>
              </Grid>
              <Grid item xs={12} md={6}>
                <Paper elevation={0} sx={{ bgcolor: '#f1f8e9', p: 2, height: '100%' }}>
                  <Typography variant="h6" gutterBottom>
                    Minimum Volatility Optimization
                  </Typography>
                  <ul>
                    <li><Typography paragraph>Focuses exclusively on minimizing risk</Typography></li>
                    <li><Typography paragraph>Does not require expected returns estimates</Typography></li>
                    <li><Typography paragraph>More robust to estimation errors</Typography></li>
                    <li><Typography paragraph>Offers a single solution (minimum risk point)</Typography></li>
                    <li><Typography paragraph>Focuses purely on risk minimization regardless of returns</Typography></li>
                  </ul>
                </Paper>
              </Grid>
            </Grid>
          </Box>
        </Paper>
        
        {/* Visual Representation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Visual Representation on the Efficient Frontier
          </Typography>
          <Typography paragraph>
            On the Efficient Frontier graph, the Minimum Volatility portfolio represents the leftmost point—the 
            portfolio with the absolute minimum variance (or standard deviation). This is where the efficient 
            frontier begins.
          </Typography>
          
          <Box sx={{ textAlign: 'center', my: 3 }}>
            {/* Placeholder for an image - in production, replace with actual image */}
            <Paper 
              elevation={0}
              sx={{ 
                width: '100%', 
                height: 300, 
                bgcolor: '#f0f0f0', 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center' 
              }}
            >
              <Typography variant="body2" color="text.secondary">
                [Minimum Volatility on Efficient Frontier Graph Placeholder]
              </Typography>
            </Paper>
            <Typography variant="caption" display="block" sx={{ mt: 1 }}>
              The Minimum Volatility portfolio (highlighted) at the leftmost point of the Efficient Frontier
            </Typography>
          </Box>
        </Paper>
        
        {/* Properties of Min-Vol Portfolios */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Properties of Minimum Volatility Portfolios
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
                      <strong>Lower Sensitivity to Estimation Errors:</strong> Does not require expected return estimates, 
                      which are often the most error-prone inputs in portfolio optimization.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Downside Protection:</strong> Typically exhibits better performance during market downturns.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Lower Drawdowns:</strong> Generally experiences smaller maximum drawdowns compared to 
                      other optimization approaches.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Diversification:</strong> Often leads to more diversified portfolios as it seeks to minimize 
                      covariance between assets.
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
                      <strong>Potentially Lower Returns:</strong> May sacrifice returns in exchange for lower risk, 
                      particularly during strong bull markets.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Concentration Risk:</strong> Without additional constraints, may concentrate in low-volatility 
                      sectors (e.g., utilities, consumer staples).
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Still Dependent on Covariance Estimates:</strong> While avoiding return forecasts, still 
                      relies on accurate covariance matrix estimation.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Single Solution:</strong> Provides only one portfolio solution rather than a spectrum of 
                      risk-return options.
                    </Typography>
                  </li>
                </ul>
              </Box>
            </Grid>
          </Grid>
        </Paper>
        
        {/* Use Cases */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Use Cases of Minimum Volatility Portfolios
          </Typography>
          
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12} md={4}>
              <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 2, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Conservative Investors
                </Typography>
                <Typography variant="body2">
                  Ideal for investors prioritizing capital preservation over growth, such as retirees or 
                  those nearing retirement who cannot afford significant drawdowns.
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 2, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Volatile Market Environments
                </Typography>
                <Typography variant="body2">
                  Particularly useful during periods of heightened market volatility, economic uncertainty, 
                  or bear markets, providing relative stability amid turbulence.
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 2, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Regulatory Constraints
                </Typography>
                <Typography variant="body2">
                  Addresses scenarios where risk management takes precedence over performance due to 
                  regulatory requirements or institutional mandates.
                </Typography>
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
                <strong>Clarke, R., De Silva, H., & Thorley, S. (2006)</strong>. "Minimum-Variance Portfolios in the U.S. Equity Market." <em>Journal of Portfolio Management</em>, 33(1), 10-24.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Haugen, R.A., & Baker, N.L. (1991)</strong>. "The Efficient Market Inefficiency of Capitalization-Weighted Stock Portfolios." <em>The Journal of Portfolio Management</em>, 17(3), 35-40.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Chan, L.K.C., Karceski, J., & Lakonishok, J. (1999)</strong>. "On Portfolio Optimization: Forecasting Covariances and Choosing the Risk Model." <em>Review of Financial Studies</em>, 12(5), 937-974.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Jagannathan, R., & Ma, T. (2003)</strong>. "Risk Reduction in Large Portfolios: Why Imposing the Wrong Constraints Helps." <em>The Journal of Finance</em>, 58(4), 1651-1683.
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
                  Mean-Variance Optimization
                </Typography>
                <Typography variant="body2" paragraph>
                  The cornerstone of Modern Portfolio Theory that balances return and risk.
                </Typography>
                <Link href="/docs/mvo" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Risk Parity
                </Typography>
                <Typography variant="body2" paragraph>
                  An alternative approach that equalizes the risk contribution from each asset in the portfolio.
                </Typography>
                <Link href="/docs/hrp" passHref>
                  <Button variant="contained" color="primary">Learn More</Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Efficient Frontier
                </Typography>
                <Typography variant="body2" paragraph>
                  The set of optimal portfolios that offer the highest expected return for a defined level of risk.
                </Typography>
                <Link href="/docs/efficient-frontier" passHref>
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

// Add getStaticProps to generate this page at build time
export const getStaticProps = async () => {
  return {
    props: {},
  };
};

export default MinimumVolatilityPage; 