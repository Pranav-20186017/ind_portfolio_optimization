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

const SortinoRatioPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Sortino Ratio | Portfolio Optimization</title>
        <meta name="description" content="Learn about the Sortino Ratio, a modification of the Sharpe ratio that only penalizes returns falling below a specified target or required rate of return." />
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
            Sortino Ratio
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            Evaluating downside risk-adjusted return in portfolio performance
          </Typography>
        </Box>
        
        {/* Overview Section */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Overview
          </Typography>
          <Typography paragraph>
            The <strong>Sortino Ratio</strong>, developed by Frank Sortino in the 1980s, refines the idea behind the Sharpe Ratio. 
            It specifically measures the risk-adjusted return of an investment by considering only <strong>downside volatility</strong>—the risk of negative returns. 
            Unlike the Sharpe Ratio, which treats all volatility as equally undesirable, the Sortino Ratio distinguishes between "good" volatility (positive returns) and "bad" volatility (negative returns), 
            thus better aligning with typical investor preferences.
          </Typography>
        </Paper>
        
        {/* Intuitive Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Intuitive Explanation
          </Typography>
          <Typography paragraph>
            Imagine you're evaluating two runners based on consistency. One occasionally runs faster than expected (positive volatility) 
            while the other frequently runs slower (negative volatility). Naturally, you'd prefer the first, as positive deviations from 
            expectations are beneficial, while negative deviations aren't.
          </Typography>
          <Typography paragraph>
            Similarly, investors prefer investments where deviations are positive (exceed expectations) rather than negative. 
            The Sortino Ratio captures this investor preference by only penalizing negative deviations, thus focusing purely on downside risk.
          </Typography>
          <Box sx={{ bgcolor: 'background.paper', p: 2, borderRadius: 1, border: '1px solid #e0e0e0', mt: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              <strong>Example:</strong>
            </Typography>
            <Typography variant="body1">
              Consider two portfolios with identical Sharpe Ratios of 1.0 and the same average return of 8%, each with a different return pattern:
            </Typography>
            <ul>
              <li>
                <Typography>
                  Portfolio A: Frequent small positive returns with occasional large negative returns
                </Typography>
              </li>
              <li>
                <Typography>
                  Portfolio B: Occasional large positive returns with frequent small negative returns
                </Typography>
              </li>
            </ul>
            <Typography variant="body1">
              While both portfolios have the same standard deviation (and thus Sharpe Ratio), Portfolio A would have a better Sortino Ratio
              because it has less downside deviation, making it more attractive to most investors who are particularly concerned with avoiding losses.
            </Typography>
          </Box>
        </Paper>
        
        {/* Mathematical Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Detailed Mathematical Explanation
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom>
            The Sortino Ratio Formula
          </Typography>
          <Typography paragraph>
            The Sortino Ratio is mathematically defined as:
          </Typography>
          <Equation math="\text{Sortino Ratio} = \frac{R_p - R_f}{\sigma_d}" />
          
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
                <InlineMath math="R_f" /> is the <strong>annualized risk-free rate</strong>.
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <InlineMath math="\sigma_d" /> is the <strong>annualized downside deviation</strong> of portfolio returns (volatility considering only negative returns).
              </Typography>
            </li>
          </ul>

          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Downside Deviation (σₘ)
          </Typography>
          <Typography paragraph>
            The crucial difference in the Sortino Ratio is its risk measure, downside deviation, defined as:
          </Typography>
          <Equation math="\sigma_d = \sqrt{\frac{1}{n}\sum_{t=1}^{n} \min(r_{p,t}-T,0)^2} \times \sqrt{252}" />
          
          <Typography paragraph>
            where:
          </Typography>
          <ul>
            <li>
              <Typography component="div">
                <InlineMath math="r_{p,t}" /> is the <strong>daily portfolio return</strong> at time t.
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <InlineMath math="T" /> is the <strong>target return</strong> (usually set to zero or the risk-free rate).
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <InlineMath math="n" /> is the <strong>number of observations</strong>.
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <InlineMath math="252" /> is used to <strong>annualize</strong> daily downside deviation (trading days per year).
              </Typography>
            </li>
          </ul>
          
          <Typography paragraph>
            Typically, the target <InlineMath math="T" /> is set to zero, capturing all negative returns.
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Implementation in Our Portfolio Optimizer
          </Typography>
          <Typography paragraph>
            Our portfolio optimization application calculates the Sortino Ratio as follows:
          </Typography>
          
          <Typography variant="subtitle1" gutterBottom>
            <strong>Implementation Logic:</strong>
          </Typography>
          <ol>
            <li>
              <Typography paragraph>
                <strong>Calculate daily returns</strong> (<InlineMath math="r_{p,t}" />)
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Calculate annualized return</strong> (<InlineMath math="R_p" />):
              </Typography>
              <Equation math="R_p = \text{mean}(r_{p,t}) \times 252" />
            </li>
            <li>
              <Typography paragraph>
                <strong>Calculate downside deviation</strong>:
              </Typography>
              <ul>
                <li>
                  <Typography>
                    Identify negative daily returns
                  </Typography>
                </li>
                <li>
                  <Typography>
                    Compute standard deviation of only these negative returns (downside deviation)
                  </Typography>
                </li>
              </ul>
            </li>
            <li>
              <Typography paragraph>
                <strong>Calculate Sortino Ratio</strong>:
              </Typography>
              <Equation math="\text{Sortino Ratio} = \frac{R_p - R_f}{\sigma_d \times \sqrt{252}}" />
            </li>
          </ol>
          
          <Box sx={{ bgcolor: '#f8f9fa', p: 3, borderRadius: 1, mt: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              <strong>Example code from our backend:</strong>
            </Typography>
            <Box component="pre" sx={{ p: 2, bgcolor: '#f1f1f1', borderRadius: 1, overflow: 'auto' }}>
              <code>
                downside_std = port_returns[port_returns {'<'} 0].std(){'\n'}
                sortino = 0.0{'\n'}
                if downside_std {'>'} 1e-9:{'\n'}
                {'    '}mean_daily = port_returns.mean(){'\n'}
                {'    '}annual_ret = mean_daily * 252{'\n'}
                {'    '}sortino = (annual_ret - risk_free_rate) / (downside_std * np.sqrt(252))
              </code>
            </Box>
          </Box>
        </Paper>
        
        {/* Why It Matters */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Why the Sortino Ratio Matters
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={4}>
              <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 2, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Investor-Centric Metric
                </Typography>
                <Typography variant="body2">
                  Emphasizes downside risk, aligning closely with typical investor preferences who are more concerned about losses than gains.
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 2, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Distinguishes Good from Bad Volatility
                </Typography>
                <Typography variant="body2">
                  Avoids penalizing portfolios that have positive, beneficial volatility, leading to more nuanced risk assessment.
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 2, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Risk Management Tool
                </Typography>
                <Typography variant="body2">
                  Especially useful during bear markets or volatile periods, providing a clearer risk-adjusted performance perspective when downside protection is particularly important.
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </Paper>
        
        {/* Interpreting Sortino Ratio */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Interpreting the Sortino Ratio
          </Typography>
          
          <Box sx={{ p: 3, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', my: 2 }}>
            <Grid container spacing={2}>
              <Grid item xs={12} md={4}>
                <Typography variant="h6" gutterBottom align="center">
                  Sortino Ratio Less Than 1
                </Typography>
                <Typography align="center" paragraph>
                  Below average performance
                </Typography>
                <Typography variant="body2">
                  Investment's returns don't sufficiently compensate for downside risk. The portfolio may have too many significant negative returns.
                </Typography>
              </Grid>
              <Grid item xs={12} md={4}>
                <Typography variant="h6" gutterBottom align="center">
                  Sortino Ratio = 1 to 2
                </Typography>
                <Typography align="center" paragraph>
                  Good performance
                </Typography>
                <Typography variant="body2">
                  Favorable downside risk-return trade-off. The portfolio demonstrates a good balance between returns and protection against losses.
                </Typography>
              </Grid>
              <Grid item xs={12} md={4}>
                <Typography variant="h6" gutterBottom align="center">
                  Sortino Ratio Greater Than 2
                </Typography>
                <Typography align="center" paragraph>
                  Excellent performance
                </Typography>
                <Typography variant="body2">
                  Very attractive investment with well-managed downside risk. The portfolio delivers strong returns while effectively limiting losses.
                </Typography>
              </Grid>
            </Grid>
            <Box sx={{ mt: 3 }}>
              <Typography variant="subtitle1" gutterBottom>
                <strong>General Rule:</strong> Higher Sortino Ratio = better downside risk-adjusted returns
              </Typography>
            </Box>
          </Box>
        </Paper>
        
        {/* Direct Comparison with Sharpe Ratio */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Sortino Ratio vs. Sharpe Ratio: Key Differences
          </Typography>
          
          <Box sx={{ mb: 3 }}>
            <Typography paragraph>
              While both ratios measure risk-adjusted returns, they differ fundamentally in how they define and calculate risk:
            </Typography>
          </Box>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 3, bgcolor: '#e3f2fd', borderRadius: 2, height: '100%' }}>
                <Typography variant="h6" gutterBottom align="center">
                  Sharpe Ratio
                </Typography>
                <ul>
                  <li>
                    <Typography paragraph>
                      <strong>Risk Measure:</strong> Total volatility (standard deviation of all returns)
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Formula Denominator:</strong> <InlineMath math="\sigma_p" /> (standard deviation of all returns)
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Penalizes:</strong> Both upside and downside volatility equally
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Assumption:</strong> Investors are concerned with overall volatility
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Best Used When:</strong> Returns are normally distributed or when both upside and downside risks matter equally
                    </Typography>
                  </li>
                </ul>
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 3, bgcolor: '#e8f5e9', borderRadius: 2, height: '100%' }}>
                <Typography variant="h6" gutterBottom align="center">
                  Sortino Ratio
                </Typography>
                <ul>
                  <li>
                    <Typography paragraph>
                      <strong>Risk Measure:</strong> Downside deviation (standard deviation of only negative returns)
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Formula Denominator:</strong> <InlineMath math="\sigma_d" /> (downside deviation)
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Penalizes:</strong> Only negative (harmful) volatility
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Assumption:</strong> Investors are primarily concerned with losing money
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Best Used When:</strong> Returns are asymmetric or when downside protection is a priority
                    </Typography>
                  </li>
                </ul>
              </Box>
            </Grid>
          </Grid>
          
          <Box sx={{ mt: 3 }}>
            <Typography variant="h6" gutterBottom>
              Numerical Example
            </Typography>
            <Typography paragraph>
              Consider a portfolio with the following characteristics:
            </Typography>
            <ul>
              <li>
                <Typography>Annual return: 12%</Typography>
              </li>
              <li>
                <Typography>Risk-free rate: 3%</Typography>
              </li>
              <li>
                <Typography>Standard deviation of all returns: 15%</Typography>
              </li>
              <li>
                <Typography>Standard deviation of only negative returns: 10%</Typography>
              </li>
            </ul>
            <Typography>
              <strong>Sharpe Ratio:</strong> (12% - 3%) / 15% = 0.60
            </Typography>
            <Typography>
              <strong>Sortino Ratio:</strong> (12% - 3%) / 10% = 0.90
            </Typography>
            <Typography paragraph sx={{ mt: 1 }}>
              This portfolio looks significantly better when evaluated using the Sortino Ratio because it has proportionally less downside risk than total risk.
            </Typography>
          </Box>
        </Paper>
        
        {/* Advantages over Sharpe Ratio */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Advantages over Sharpe Ratio
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, bgcolor: '#e8f5e9', borderRadius: 2, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Investor Focused
                </Typography>
                <Typography variant="body2">
                  More closely aligns with investor concerns, emphasizing losses over volatility in general. This psychological alignment with how investors actually perceive risk makes the metric more intuitive.
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, bgcolor: '#e8f5e9', borderRadius: 2, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Better in Volatile Markets
                </Typography>
                <Typography variant="body2">
                  Provides clearer insights during market downturns or heightened volatility. When markets become turbulent, the Sortino Ratio helps distinguish between portfolios that maintain downside protection versus those that don't.
                </Typography>
              </Box>
            </Grid>
          </Grid>
          
          <Box sx={{ p: 3, bgcolor: '#f5f5f5', borderRadius: 1, mt: 3 }}>
            <Typography variant="subtitle1" gutterBottom>
              <strong>Practical Example:</strong>
            </Typography>
            <Typography variant="body2">
              In a year where the market has significant upswings and downswings, two portfolios might have identical Sharpe Ratios. 
              However, the portfolio that captured more of the upswings while avoiding downswings would have a higher Sortino Ratio, 
              correctly reflecting its superior risk management approach from an investor's perspective.
            </Typography>
          </Box>
        </Paper>
        
        {/* Limitations */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Limitations
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom color="error">
                  Sensitive to Threshold Choice
                </Typography>
                <Typography variant="body2">
                  The choice of threshold (usually zero or risk-free rate) can significantly affect results. Different threshold choices can lead to different rankings of portfolios, potentially leading to inconsistent evaluations.
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom color="error">
                  Statistical Significance
                </Typography>
                <Typography variant="body2">
                  Requires enough negative observations to accurately estimate downside deviation. With limited data or in strong bull markets, there may be too few negative returns to reliably calculate the ratio.
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
                <strong>Sortino, F. A., & Price, L. N. (1994)</strong>. "Performance Measurement in a Downside Risk Framework." <em>Journal of Investing</em>, 3(3), 59-64.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Sortino, F. A., & Van Der Meer, R. (1991)</strong>. "Downside Risk." <em>Journal of Portfolio Management</em>, 17(4), 27-31.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Sortino, F. A., & Satchell, S. (2001)</strong>. <em>Managing Downside Risk in Financial Markets</em>. Butterworth-Heinemann.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Kaplan, P. D., & Knowles, J. A. (2004)</strong>. "Kappa: A Generalized Downside Risk-Adjusted Performance Measure." <em>Journal of Performance Measurement</em>, 8, 42-54.
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
                  Sharpe Ratio
                </Typography>
                <Typography variant="body2" paragraph>
                  The precursor to the Sortino Ratio that measures excess return per unit of total volatility.
                </Typography>
                <Link href="/education/sharpe-ratio" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Omega Ratio
                </Typography>
                <Typography variant="body2" paragraph>
                  A performance metric that evaluates the probability-weighted ratio of gains to losses for a threshold return.
                </Typography>
                <Button variant="outlined" color="primary" disabled>Coming Soon</Button>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Calmar Ratio
                </Typography>
                <Typography variant="body2" paragraph>
                  A risk-adjusted performance measure that relates average annual compound returns to maximum drawdown.
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

export default SortinoRatioPage; 