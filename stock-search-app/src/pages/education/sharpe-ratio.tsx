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

const SharpeRatioPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Sharpe Ratio | Portfolio Optimization</title>
        <meta name="description" content="Learn about the Sharpe Ratio, a measure of risk-adjusted return that helps investors understand the return of an investment compared to its risk." />
      </Head>
      
      <Container maxWidth="lg" sx={{ py: 4 }}>
        {/* Navigation Buttons */}
        <Box sx={{ mb: 4, display: 'flex', gap: 2 }}>
          <Link href="/education" passHref>
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
            Sharpe Ratio
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            Measuring risk-adjusted performance in portfolio optimization
          </Typography>
        </Box>
        
        {/* Overview Section */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Overview
          </Typography>
          <Typography paragraph>
            The <strong>Sharpe Ratio</strong>, developed by Nobel laureate William F. Sharpe in 1966, is one of the most popular and widely used metrics in finance. 
            It provides a clear measure of risk-adjusted performance by evaluating the returns of an investment relative to its volatility. 
            Specifically, the Sharpe Ratio helps investors understand whether the returns they are achieving justify the level of risk taken.
          </Typography>
        </Paper>
        
        {/* Intuitive Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Intuitive Explanation
          </Typography>
          <Typography paragraph>
            Imagine you're comparing two cars based on fuel efficiency (distance traveled per unit of fuel). A more fuel-efficient car gives you more miles per gallon. 
            Similarly, the Sharpe Ratio measures investment "efficiency," assessing how much return an investment provides per unit of risk taken.
          </Typography>
          <Typography paragraph>
            A higher Sharpe Ratio indicates a better return for each unit of risk, making the investment more attractive, while a lower Sharpe Ratio suggests 
            insufficient returns given the risk involved.
          </Typography>
          <Box sx={{ bgcolor: 'background.paper', p: 2, borderRadius: 1, border: '1px solid #e0e0e0', mt: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              <strong>Example:</strong>
            </Typography>
            <Typography variant="body1">
              Consider two portfolios, both with an annual return of 10%:
            </Typography>
            <ul>
              <li>
                <Typography>
                  Portfolio A has a volatility (standard deviation) of 15% and a risk-free rate of 2%, giving a Sharpe Ratio of (10% - 2%) / 15% = 0.53
                </Typography>
              </li>
              <li>
                <Typography>
                  Portfolio B has a volatility of 8% and the same risk-free rate, giving a Sharpe Ratio of (10% - 2%) / 8% = 1.00
                </Typography>
              </li>
            </ul>
            <Typography variant="body1">
              Portfolio B is significantly more efficient at generating returns for each unit of risk taken, making it the better choice despite both portfolios having the same absolute return.
            </Typography>
          </Box>
        </Paper>
        
        {/* Mathematical Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Detailed Mathematical Explanation
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom>
            The Sharpe Ratio Formula
          </Typography>
          <Typography paragraph>
            The mathematical definition of the Sharpe Ratio is straightforward and intuitive:
          </Typography>
          <Equation math="\text{Sharpe Ratio} = \frac{R_p - R_f}{\sigma_p}" />
          
          <Typography paragraph>
            where:
          </Typography>
          <ul>
            <li>
              <Typography component="div">
                <InlineMath math="R_p" /> is the <strong>annualized expected return</strong> of the portfolio.
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <InlineMath math="R_f" /> is the <strong>annualized risk-free rate</strong> (usually government treasury yield).
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <InlineMath math="\sigma_p" /> is the <strong>annualized standard deviation</strong> (volatility) of the portfolio returns.
              </Typography>
            </li>
          </ul>

          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Interpretation of Formula
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                The numerator <InlineMath math="(R_p - R_f)" /> is the <strong>"excess return,"</strong> representing how much more the investment returns compared to a risk-free investment.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                The denominator <InlineMath math="\sigma_p" /> captures the volatility or riskiness of the investment.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                The ratio directly compares reward (returns) to risk (volatility).
              </Typography>
            </li>
          </ul>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Implementation in Our Portfolio Optimizer
          </Typography>
          <Typography paragraph>
            Our portfolio optimization application calculates the Sharpe Ratio explicitly using annualized metrics:
          </Typography>
          
          <Typography variant="subtitle1" gutterBottom>
            <strong>Implementation Logic:</strong>
          </Typography>
          <ol>
            <li>
              <Typography paragraph>
                <strong>Compute daily returns:</strong>
              </Typography>
              <Equation math="r_{p,t} = \text{Portfolio daily returns}" />
            </li>
            <li>
              <Typography paragraph>
                <strong>Annualize returns and volatility:</strong>
              </Typography>
              <Typography paragraph>
                Annualized return:
              </Typography>
              <Equation math="R_p = \text{mean}(r_{p,t}) \times 252" />
              <Typography paragraph>
                Annualized volatility:
              </Typography>
              <Equation math="\sigma_p = \text{std}(r_{p,t}) \times \sqrt{252}" />
              <Typography variant="caption">
                (Assuming 252 trading days in a year.)
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Risk-free rate</strong> (<InlineMath math="R_f" />):
                Typically obtained from treasury bill yields or other safe investment benchmarks (annualized).
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Calculate Sharpe Ratio:</strong>
              </Typography>
              <Equation math="\text{Sharpe Ratio} = \frac{R_p - R_f}{\sigma_p}" />
            </li>
          </ol>
          
          <Box sx={{ bgcolor: '#f8f9fa', p: 3, borderRadius: 1, mt: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              <strong>Example code from our backend:</strong>
            </Typography>
            <Box component="pre" sx={{ p: 2, bgcolor: '#f1f1f1', borderRadius: 1, overflow: 'auto' }}>
              <code>
                annual_return = port_returns.mean() * 252{'\n'}
                annual_volatility = port_returns.std() * np.sqrt(252){'\n'} 
                sharpe = (annual_return - risk_free_rate) / annual_volatility if annual_volatility {'>'} 0 else 0.0
              </code>
            </Box>
          </Box>
        </Paper>
        
        {/* Why It Matters */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Why the Sharpe Ratio Matters
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={4}>
              <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 2, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Risk-adjusted Comparisons
                </Typography>
                <Typography variant="body2">
                  Allows comparing assets or strategies with varying levels of risk, putting investments on equal footing regardless of their absolute risk levels.
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 2, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Portfolio Selection
                </Typography>
                <Typography variant="body2">
                  Investors often choose portfolios with higher Sharpe Ratios, maximizing returns for a given level of risk. In portfolio theory, the optimal portfolio often maximizes the Sharpe Ratio.
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 2, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Performance Evaluation
                </Typography>
                <Typography variant="body2">
                  It's a critical metric for evaluating fund managers and investment strategies, helping to determine if higher returns are due to skill or simply from taking higher risks.
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </Paper>
        
        {/* Interpreting Sharpe Ratio */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Interpreting the Sharpe Ratio
          </Typography>
          
          <Box sx={{ p: 3, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', my: 2 }}>
            <Grid container spacing={2}>
              <Grid item xs={12} md={4}>
                <Typography variant="h6" gutterBottom align="center">
                  Sharpe Ratio Less Than 1
                </Typography>
                <Typography align="center" paragraph>
                  Below average performance
                </Typography>
                <Typography variant="body2">
                  Investment may not adequately compensate for the risk taken. Returns may be too low for the volatility experienced.
                </Typography>
              </Grid>
              <Grid item xs={12} md={4}>
                <Typography variant="h6" gutterBottom align="center">
                  Sharpe Ratio = 1 to 2
                </Typography>
                <Typography align="center" paragraph>
                  Good performance
                </Typography>
                <Typography variant="body2">
                  Balanced risk-return tradeoff. The portfolio is generating a reasonable excess return for the risk taken.
                </Typography>
              </Grid>
              <Grid item xs={12} md={4}>
                <Typography variant="h6" gutterBottom align="center">
                  Sharpe Ratio Greater Than 2
                </Typography>
                <Typography align="center" paragraph>
                  Excellent performance
                </Typography>
                <Typography variant="body2">
                  Very attractive risk-adjusted returns. The investment is generating substantial returns relative to its volatility.
                </Typography>
              </Grid>
            </Grid>
            <Box sx={{ mt: 3 }}>
              <Typography variant="subtitle1" gutterBottom>
                <strong>Key Principle:</strong> Higher is better - indicates more efficient risk-taking
              </Typography>
            </Box>
          </Box>
        </Paper>
        
        {/* Limitations */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Limitations of the Sharpe Ratio
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom color="error">
                  Normality Assumption
                </Typography>
                <Typography variant="body2">
                  Assumes returns follow a normal distribution, which might not hold true. Financial returns often have fat tails and skewness, making extreme events more likely than the normal distribution suggests.
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom color="error">
                  Sensitive to Outliers
                </Typography>
                <Typography variant="body2">
                  Extreme returns significantly affect the Sharpe Ratio. A single quarter of exceptional performance or poor performance can substantially alter the ratio, potentially masking the typical performance.
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom color="error">
                  Symmetric View of Risk
                </Typography>
                <Typography variant="body2">
                  Treats positive and negative volatility equally, though investors typically care more about downside volatility. This is why metrics like the Sortino Ratio were developed to focus specifically on downside risk.
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
                <strong>Sharpe, W. F. (1966)</strong>. "Mutual Fund Performance." <em>The Journal of Business</em>, 39(1), 119-138.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Sharpe, W. F. (1994)</strong>. "The Sharpe Ratio." <em>The Journal of Portfolio Management</em>, 21(1), 49-58.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Lo, A. W. (2002)</strong>. "The Statistics of Sharpe Ratios." <em>Financial Analysts Journal</em>, 58(4), 36-52.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Bailey, D. H., & Lopez de Prado, M. (2012)</strong>. "The Sharpe Ratio Efficient Frontier." <em>Journal of Risk</em>, 15(2), 3-44.
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
                  Sortino Ratio
                </Typography>
                <Typography variant="body2" paragraph>
                  A variation of the Sharpe Ratio that only penalizes downside volatility, focusing on harmful risk.
                </Typography>
                <Link href="/education/sortino-ratio" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Volatility (σ)
                </Typography>
                <Typography variant="body2" paragraph>
                  A statistical measure of the dispersion of returns that is a key component in calculating the Sharpe Ratio.
                </Typography>
                <Link href="/education/volatility" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Treynor Ratio
                </Typography>
                <Typography variant="body2" paragraph>
                  Similar to Sharpe but uses beta (systematic risk) instead of standard deviation, measuring excess return per unit of market risk.
                </Typography>
                <Button variant="outlined" color="primary" disabled>Coming Soon</Button>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Information Ratio
                </Typography>
                <Typography variant="body2" paragraph>
                  Measures the risk-adjusted returns of a portfolio relative to a benchmark, useful for evaluating active management.
                </Typography>
                <Button variant="outlined" color="primary" disabled>Coming Soon</Button>
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

export default SharpeRatioPage; 