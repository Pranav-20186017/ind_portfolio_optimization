import React from 'react';
import {
  Typography,
  Container,
  Box,
  Paper,
  Grid,
  Button,
  Link as MuiLink
} from '@mui/material';
import Link from 'next/link';
import Head from 'next/head';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';

// Re‑usable equation wrapper for consistent styling
const Equation = ({ math }: { math: string }) => (
  <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 1, my: 2, textAlign: 'center' }}>
    <BlockMath math={math} />
  </Box>
);

const InformationRatioPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Information Ratio | Portfolio Optimization</title>
        <meta
          name="description"
          content="Learn about the Information Ratio, a performance metric that evaluates active return per unit of risk relative to a benchmark index."
        />
      </Head>

      <Container maxWidth="lg" sx={{ py: 4 }}>
        {/* Navigation */}
        <Box sx={{ mb: 4, display: 'flex', gap: 2 }}>
          <Link href="/docs" passHref>
            <Button variant="outlined" color="primary">← Back to Education</Button>
          </Link>
          <Link href="/" passHref>
            <Button variant="outlined" color="secondary">← Back to Portfolio Optimizer</Button>
          </Link>
        </Box>

        {/* Title */}
        <Box sx={{ mb: 6, textAlign: 'center' }}>
          <Typography variant="h3" component="h1" gutterBottom>
            Information Ratio
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            A performance metric that evaluates active return per unit of risk relative to a benchmark index
          </Typography>
        </Box>

        {/* Overview */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Overview</Typography>
          <Typography paragraph>
            <strong>Information Ratio (IR)</strong> is a key performance metric used to evaluate the skill of active portfolio managers or investment strategies. It measures how much excess return (active return) a portfolio manager generates relative to a benchmark, per unit of additional risk (tracking error) taken.
          </Typography>
          <Typography paragraph>
            The Information Ratio provides insight into a manager's ability to generate consistent outperformance through active decisions, while accounting for the additional risk assumed in deviating from the benchmark. This makes it particularly valuable for institutional investors and asset allocators evaluating active management capabilities.
          </Typography>
          <Typography paragraph>
            Unlike the Sharpe ratio, which measures excess return over the risk-free rate per unit of total risk, the Information Ratio focuses specifically on active management decisions by measuring excess return over a benchmark per unit of active risk. This distinction is crucial for isolating and evaluating the true value added by active management.
          </Typography>
        </Paper>

        {/* Intuitive Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Intuitive Explanation</Typography>
          <Typography paragraph>
            Imagine you've hired two different guides to take you through a forest. Both guides deviate from the main trail (the benchmark) in search of better views or shortcuts:
          </Typography>
          <Typography paragraph>
            <strong>Guide A</strong> takes you on paths that frequently offer slightly better views than the main trail, with minimal additional hiking difficulty.
          </Typography>
          <Typography paragraph>
            <strong>Guide B</strong> takes you on paths that occasionally offer spectacular views, but also involve significant additional climbing and rough terrain.
          </Typography>
          <Typography paragraph>
            The Information Ratio helps determine which guide provides better value. Guide A might have a higher Information Ratio because they consistently deliver modest improvements with minimal extra effort. Guide B might provide occasional amazing experiences but with inconsistent results and much more exertion.
          </Typography>
          <Box sx={{ bgcolor: 'background.paper', p: 2, borderRadius: 1, border: '1px solid #e0e0e0', mt: 2 }}>
            <Typography variant="body1">
              <strong>Financial analogy:</strong> In investing terms, the Information Ratio rewards consistent, reliable outperformance over a benchmark relative to the additional risk taken. A high Information Ratio indicates a manager who efficiently converts active risk (deviations from the benchmark) into active return (outperformance), demonstrating skill rather than luck.
            </Typography>
          </Box>
        </Paper>

        {/* Mathematical Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Detailed Mathematical Explanation</Typography>

          <Typography paragraph>
            The Information Ratio quantifies the relationship between the active return of a portfolio and the active risk taken to achieve that return.
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>Core Formula</Typography>
          <Typography paragraph>
            The Information Ratio is defined as:
          </Typography>

          <Box sx={{ p: 3, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', my: 3 }}>
            <Typography variant="subtitle1" gutterBottom><strong>Information Ratio Formula</strong></Typography>
            <Equation math="IR = \frac{\text{Active Return}}{\text{Active Risk}} = \frac{R_p - R_b}{\sigma_{p-b}}" />
            <Typography variant="body2">
              where Active Return is the portfolio return minus the benchmark return, and Active Risk (tracking error) is the standard deviation of the active return.
            </Typography>
          </Box>

          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>Formal Definition</Typography>
          <Typography paragraph>
            More precisely, the Information Ratio is calculated as:
          </Typography>

          <Equation math="IR = \frac{R_p - R_b}{\sqrt{\frac{1}{T}\sum_{t=1}^{T}[(R_{p,t} - R_{b,t}) - (R_p - R_b)]^2}}" />

          <Typography paragraph>
            Where:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <InlineMath math="R_p" /> is the average return of the portfolio
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <InlineMath math="R_b" /> is the average return of the benchmark
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <InlineMath math="R_{p,t}" /> is the return of the portfolio at time t
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <InlineMath math="R_{b,t}" /> is the return of the benchmark at time t
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <InlineMath math="\sigma_{p-b}" /> is the tracking error (standard deviation of the difference between portfolio and benchmark returns)
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <InlineMath math="T" /> is the number of observation periods
              </Typography>
            </li>
          </ul>

          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>Alternative Formulation</Typography>
          <Typography paragraph>
            The Information Ratio can also be expressed in terms of active returns:
          </Typography>

          <Box sx={{ p: 3, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', my: 3 }}>
            <Typography variant="subtitle1" gutterBottom><strong>Information Ratio using Active Returns</strong></Typography>
            <Equation math="IR = \frac{\alpha}{\sigma_{\alpha}}" />
            <Typography variant="body2">
              where <InlineMath math="\alpha" /> represents the active return and <InlineMath math="\sigma_{\alpha}" /> is the standard deviation of active returns (tracking error).
            </Typography>
          </Box>

          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>Relationship to Other Metrics</Typography>
          <Typography paragraph>
            The Information Ratio relates to other performance metrics as follows:
          </Typography>

          <ul>
            <li>
              <Typography paragraph>
                <strong>Sharpe Ratio:</strong> <InlineMath math="\text{Sharpe Ratio} = \frac{R_p - R_f}{\sigma_p}" /> uses risk-free rate as benchmark and total volatility as risk measure
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Information Ratio:</strong> <InlineMath math="\text{Information Ratio} = \frac{R_p - R_b}{\sigma_{p-b}}" /> uses market index as benchmark and tracking error as risk measure
              </Typography>
            </li>
          </ul>

          <Typography paragraph>
            The key difference is that the Information Ratio focuses specifically on active management decisions relative to a benchmark, while the Sharpe ratio measures total risk-adjusted performance regardless of benchmark.
          </Typography>
        </Paper>

        {/* Implementation Details */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Implementation in Portfolio Analysis</Typography>
          <Typography paragraph>
            Our implementation of the Information Ratio involves the following steps:
          </Typography>
          <ol>
            <li>
              <Typography paragraph>
                <strong>Define the Benchmark:</strong> We select an appropriate benchmark that represents the investment universe and risk exposures relevant to the portfolio. Common choices include:
              </Typography>
              <ul>
                <li>
                  <Typography paragraph>
                    Broad market indices (e.g., S&P 500, Russell 3000)
                  </Typography>
                </li>
                <li>
                  <Typography paragraph>
                    Style-specific indices (e.g., Russell 1000 Growth, MSCI Value)
                  </Typography>
                </li>
                <li>
                  <Typography paragraph>
                    Custom composite benchmarks to match portfolio allocation
                  </Typography>
                </li>
              </ul>
            </li>
            <li>
              <Typography paragraph>
                <strong>Calculate Active Returns:</strong> We compute the difference between portfolio returns and benchmark returns for each period.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Determine Tracking Error:</strong> We calculate the standard deviation of these active returns over the measurement period.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Compute the Ratio:</strong> We divide the average active return by the tracking error.
              </Typography>
            </li>
          </ol>
          <Typography paragraph>
            In portfolio evaluation, we use the Information Ratio to:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Assess Manager Skill:</strong> Identifying fund managers with consistent outperformance per unit of active risk.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Portfolio Construction:</strong> Allocating more capital to strategies with higher Information Ratios.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Risk Budgeting:</strong> Determining how much active risk to take in different parts of a portfolio based on expected Information Ratios.
              </Typography>
            </li>
          </ul>
        </Paper>

        {/* Worked Example */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Worked Example</Typography>
          <Typography paragraph>
            Let's calculate and compare the Information Ratio for two hypothetical fund managers relative to their benchmark.
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 1: Historical Returns Data</Typography>
          <Typography paragraph>
            Suppose we have the following quarterly returns for two active managers and their benchmark over a 3-year period (12 quarters):
          </Typography>
          <Box sx={{ p: 2, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', my: 2 }}>
            <Typography variant="body2">
              <strong>Benchmark:</strong> 2.1%, -1.5%, 3.0%, 1.8%, 2.5%, -2.0%, 4.1%, 0.9%, 3.2%, -1.1%, 2.7%, 1.6%
            </Typography>
            <Typography variant="body2" sx={{ mt: 1 }}>
              <strong>Manager A:</strong> 2.5%, -1.2%, 3.8%, 1.5%, 3.2%, -1.6%, 4.7%, 1.5%, 3.0%, -1.5%, 3.5%, 2.0%
            </Typography>
            <Typography variant="body2" sx={{ mt: 1 }}>
              <strong>Manager B:</strong> 3.1%, -2.5%, 5.0%, 0.5%, 3.5%, -1.0%, 6.1%, -0.5%, 5.2%, -3.0%, 4.2%, 3.0%
            </Typography>
          </Box>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 2: Calculate Active Returns</Typography>
          <Typography paragraph>
            First, we calculate the active returns (portfolio return minus benchmark return) for each quarter:
          </Typography>
          <Grid container spacing={3} sx={{ mb: 2 }}>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle1" gutterBottom><strong>Manager A Active Returns</strong></Typography>
              <Typography paragraph>
                0.4%, 0.3%, 0.8%, -0.3%, 0.7%, 0.4%, 0.6%, 0.6%, -0.2%, -0.4%, 0.8%, 0.4%
              </Typography>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle1" gutterBottom><strong>Manager B Active Returns</strong></Typography>
              <Typography paragraph>
                1.0%, -1.0%, 2.0%, -1.3%, 1.0%, 1.0%, 2.0%, -1.4%, 2.0%, -1.9%, 1.5%, 1.4%
              </Typography>
            </Grid>
          </Grid>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 3: Calculate Average Active Return and Tracking Error</Typography>
          <Typography paragraph>
            Now we calculate the mean active return and the standard deviation of active returns (tracking error):
          </Typography>
          
          <Typography variant="subtitle2" gutterBottom><strong>For Manager A:</strong></Typography>
          <Typography paragraph>
            Average Active Return = (0.4 + 0.3 + 0.8 + ... + 0.4) / 12 = 3.7 / 12 = 0.31%
          </Typography>
          <Typography paragraph>
            Tracking Error = Standard Deviation of Active Returns = 0.43%
          </Typography>
          
          <Typography variant="subtitle2" sx={{ mt: 2 }} gutterBottom><strong>For Manager B:</strong></Typography>
          <Typography paragraph>
            Average Active Return = (1.0 + (-1.0) + 2.0 + ... + 1.4) / 12 = 6.3 / 12 = 0.53%
          </Typography>
          <Typography paragraph>
            Tracking Error = Standard Deviation of Active Returns = 1.44%
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 4: Calculate Information Ratio</Typography>
          <Typography paragraph>
            Using our formula IR = Average Active Return / Tracking Error:
          </Typography>
          <Box sx={{ p: 2, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', my: 2 }}>
            <Typography variant="body2">
              <strong>Manager A:</strong> IR = 0.31% / 0.43% = 0.72
            </Typography>
            <Typography variant="body2" sx={{ mt: 1 }}>
              <strong>Manager B:</strong> IR = 0.53% / 1.44% = 0.37
            </Typography>
          </Box>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 5: Interpretation</Typography>
          <Typography paragraph>
            While Manager B generated a higher average active return (0.53% vs. 0.31%), Manager A has a much higher Information Ratio (0.72 vs. 0.37). This indicates that Manager A is more skilled at generating consistent outperformance per unit of risk taken.
          </Typography>
          <Typography paragraph>
            Manager A's returns show lower but more consistent outperformance with less tracking error, while Manager B takes larger active bets with higher volatility in results. For risk-conscious investors seeking reliable alpha, Manager A would be the preferred choice despite the lower absolute outperformance.
          </Typography>
          <Typography paragraph>
            This example illustrates how the Information Ratio helps investors distinguish between managers based on skill (consistent outperformance) rather than just raw returns or willingness to deviate dramatically from the benchmark.
          </Typography>
        </Paper>

        {/* Practical Applications */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Practical Applications</Typography>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Manager Selection</Typography>
          <Typography paragraph>
            The Information Ratio is extensively used in the institutional investment community for evaluating and selecting active managers. Higher Information Ratios indicate managers who more efficiently convert active risk into active return, suggesting greater skill rather than luck or excessive risk-taking.
          </Typography>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Performance Evaluation</Typography>
          <Typography paragraph>
            By focusing on risk-adjusted active returns, the Information Ratio provides a more comprehensive assessment of manager performance than simple outperformance metrics. It helps distinguish between managers who achieve outperformance through skill versus those who rely on taking outsized risks.
          </Typography>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Portfolio Construction</Typography>
          <Typography paragraph>
            In a multi-manager portfolio, capital can be allocated to different strategies based on their historical Information Ratios, with more capital assigned to managers with higher ratios. This approach optimizes the overall portfolio's active risk-return profile.
          </Typography>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Risk Budgeting</Typography>
          <Typography paragraph>
            Investment committees use the Information Ratio to allocate active risk budgets across different portfolio segments. Areas with historically higher Information Ratios may be granted larger risk budgets, as they have demonstrated better conversion of risk into return.
          </Typography>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Fee Evaluation</Typography>
          <Typography paragraph>
            The Information Ratio helps investors assess whether the higher fees charged by active managers are justified by their skill in generating risk-adjusted outperformance. Managers with consistently high Information Ratios may warrant higher fees than those with lower ratios.
          </Typography>
        </Paper>

        {/* Advantages and Limitations */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Advantages and Limitations</Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>              
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>                
                <Typography variant="h6" color="primary" gutterBottom>Advantages</Typography>                
                <ul>                  
                  <li>                    
                    <Typography paragraph>                      
                      <strong>Focus on Active Management:</strong> Directly measures the value added by active decisions, isolating manager skill from market movements.                    
                    </Typography>                  
                  </li>                  
                  <li>                    
                    <Typography paragraph>                      
                      <strong>Risk Adjustment:</strong> Accounts for the additional risk taken to achieve outperformance, not just the magnitude of returns.                    
                    </Typography>                  
                  </li>                  
                  <li>                    
                    <Typography paragraph>                      
                      <strong>Consistency Evaluation:</strong> Rewards consistent outperformance over sporadic high returns, aligning with institutional investors' preference for reliability.                    
                    </Typography>                  
                  </li>                  
                  <li>                    
                    <Typography paragraph>                      
                      <strong>Comparability:</strong> Provides a standardized way to compare managers across different strategies and market environments.                    
                    </Typography>                  
                  </li>                
                </ul>              
              </Box>            
            </Grid>                        
            <Grid item xs={12} md={6}>              
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>                
                <Typography variant="h6" color="error" gutterBottom>Limitations</Typography>                
                <ul>                  
                  <li>                    
                    <Typography paragraph>                      
                      <strong>Benchmark Sensitivity:</strong> Results are highly dependent on the choice of benchmark, which may not always perfectly represent the investment universe.                    
                    </Typography>                  
                  </li>                  
                  <li>                    
                    <Typography paragraph>                      
                      <strong>Assumes Normal Distribution:</strong> Like many traditional risk metrics, the Information Ratio implicitly assumes that returns are normally distributed, which may not hold true in practice.                    
                    </Typography>                  
                  </li>                  
                  <li>                    
                    <Typography paragraph>                      
                      <strong>Time Period Dependency:</strong> Information Ratios can vary significantly across different time periods, making it important to evaluate over multiple market cycles.                    
                    </Typography>                  
                  </li>                  
                  <li>                    
                    <Typography paragraph>                      
                      <strong>Ignores Higher Moments:</strong> Doesn't account for skewness or kurtosis in return distributions, which can be important risk factors.                    
                    </Typography>                  
                  </li>                  
                  <li>                    
                    <Typography paragraph>                      
                      <strong>No Absolute Risk Consideration:</strong> Focuses solely on relative risk (tracking error) rather than absolute risk, potentially overlooking total portfolio risk exposure.                    
                    </Typography>                  
                  </li>                
                </ul>              
              </Box>            
            </Grid>
          </Grid>
        </Paper>

        {/* References */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>References</Typography>
          <ul>
            <li>
              <Typography paragraph>
                Goodwin, T. H. (1998). "The Information Ratio." Financial Analysts Journal, 54(4), 34-43.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Grinold, R. C., & Kahn, R. N. (2000). "Active Portfolio Management: A Quantitative Approach for Providing Superior Returns and Controlling Risk." McGraw-Hill.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Gupta, F., Prajogi, R., & Stubbs, E. (1999). "The Information Ratio and Performance." Journal of Portfolio Management, 26(1), 33-39.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Sharpe, W. F. (1994). "The Sharpe Ratio." Journal of Portfolio Management, 21(1), 49-58.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Amenc, N., & Le Sourd, V. (2003). "Portfolio Theory and Performance Analysis." Wiley Finance.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Treynor, J. L., & Black, F. (1973). "How to Use Security Analysis to Improve Portfolio Selection." Journal of Business, 46(1), 66-86.
              </Typography>
            </li>
          </ul>
        </Paper>

        {/* Related Metrics */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Related Metrics
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Sharpe Ratio
                </Typography>
                <Typography variant="body2" paragraph>
                  The classic risk‐adjusted return metric that divides excess portfolio return by total volatility.
                </Typography>
                <Link href="/docs/sharpe-ratio" passHref>
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
                  A portfolio performance metric that measures returns earned in excess of the risk-free rate per unit of market risk (beta).
                </Typography>
                <Link href="/docs/treynor-ratio" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>

            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Modigliani Risk-Adjusted Performance (M²)
                </Typography>
                <Typography variant="body2" paragraph>
                  A measure that adjusts portfolio returns to match market volatility, allowing direct comparison with benchmark returns.
                </Typography>
                <Link href="/docs/modigliani-risk-adjusted" passHref>
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

export default InformationRatioPage; 