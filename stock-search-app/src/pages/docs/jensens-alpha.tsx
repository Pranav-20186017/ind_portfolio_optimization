import React from 'react';
import {
  Typography,
  Container,
  Box,
  Paper,
  Grid,
  Button,
  Link as MuiLink,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow
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

const JensensAlphaPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Jensen's Alpha for Indian Portfolios | QuantPort India Docs</title>
        <meta name="description" content="Measure risk-adjusted outperformance of Indian stock portfolios with Jensen's Alpha. Evaluate NSE/BSE investment managers and strategies by quantifying returns beyond market exposure." />
        <meta property="og:title" content="Jensen's Alpha for Indian Portfolios | QuantPort India Docs" />
        <meta property="og:description" content="Measure risk-adjusted outperformance of Indian stock portfolios with Jensen's Alpha. Evaluate NSE/BSE investment managers and strategies by quantifying returns beyond market exposure." />
        <meta property="og:type" content="article" />
        <meta property="og:url" content="https://indportfoliooptimization.vercel.app/docs/jensens-alpha" />
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
            Jensen's Alpha (α)
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            Measuring risk-adjusted outperformance beyond market exposure
          </Typography>
        </Box>
        
        {/* What Is Jensen's Alpha */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            What Is Jensen's Alpha?
          </Typography>
          <Typography paragraph>
            <strong>Jensen's Alpha</strong> measures the <em>excess return</em> an investment earns <strong>after</strong> accounting 
            for the risk it takes relative to the market, as prescribed by the <Link href="/docs/capm" passHref>
            <MuiLink>Capital Asset Pricing Model (CAPM)</MuiLink></Link>.
          </Typography>
          <Typography paragraph>
            Put differently, alpha tells you whether the manager (or strategy) has delivered <em>skill-based</em> performance beyond
            what <Link href="/docs/capm-beta" passHref><MuiLink>β</MuiLink></Link>-driven market exposure explains.
          </Typography>
          
          <Grid container spacing={2} sx={{ mt: 2 }}>
            <Grid item xs={12} sm={6}>
              <Box sx={{ p: 2, border: '1px solid #4caf50', borderRadius: 1, bgcolor: '#e8f5e9' }}>
                <Typography align="center">
                  <strong>Positive α</strong> → outperformance (value added)
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6}>
              <Box sx={{ p: 2, border: '1px solid #f44336', borderRadius: 1, bgcolor: '#ffebee' }}>
                <Typography align="center">
                  <strong>Negative α</strong> → underperformance (value destroyed)
                </Typography>
              </Box>
            </Grid>
          </Grid>

          <Box sx={{ mt: 3 }}>
            <Link href="/docs/capm" passHref>
              <Button variant="text" color="primary">
                Learn (or review) the full CAPM theory →
              </Button>
            </Link>
          </Box>
        </Paper>
        
        {/* CAPM Refresher */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            CAPM Refresher
          </Typography>
          <Typography paragraph>
            Jensen's Alpha is derived from the <Link href="/docs/capm" passHref><MuiLink>Capital Asset Pricing Model (CAPM)</MuiLink></Link> regression equation:
          </Typography>
          <Equation math="\boxed{\;R_i - R_f = \alpha_i + \beta_i\,(R_m - R_f) + \varepsilon_i\;}" />
          
          <Typography paragraph>
            Solving that regression for <InlineMath math="\alpha_i" /> and <InlineMath math="\beta_i" />:
          </Typography>
          <Equation math="\alpha_i = (R_i - R_f) - \beta_i\,(R_m - R_f)" />
          
          <Typography paragraph>
            This formula shows that alpha is the excess return of an asset or portfolio beyond what would be predicted by its market risk (beta).
          </Typography>
        </Paper>
        
        {/* Econometric Estimation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Econometric Estimation (OLS Matrix Form)
          </Typography>
          <Typography paragraph>
            Using <InlineMath math="n" /> observations of daily or monthly <strong>excess</strong> returns:
          </Typography>
          
          <Equation math="y_t = R_{i,t}-R_{f,t},\quad x_t = R_{m,t}-R_{f,t}" />
          
          <Typography paragraph>
            The regression model in matrix form is:
          </Typography>
          
          <Equation math="\underbrace{\mathbf{y}}_{n\times1} = \underbrace{\mathbf{X}}_{n\times2} \underbrace{\boldsymbol{\theta}}_{2\times1} + \boldsymbol{\varepsilon}, \qquad \mathbf{X} = \begin{bmatrix} 1 & x_1\\ 1 & x_2\\ \vdots & \vdots\\ 1 & x_n \end{bmatrix}, \; \boldsymbol{\theta} = \begin{bmatrix}\alpha\\\beta\end{bmatrix}" />
          
          <Typography paragraph>
            The Ordinary Least Squares (OLS) solution is:
          </Typography>
          
          <Equation math="\boxed{\; \hat{\boldsymbol{\theta}} = (\mathbf{X}^{\!\top}\mathbf{X})^{-1}\mathbf{X}^{\!\top}\mathbf{y} \;}" />
          
          <Box sx={{ mt: 2, ml: 4 }}>
            <Typography component="div">
              • <InlineMath math="\hat{\alpha}" /> — Jensen's Alpha (annualized in practice)
            </Typography>
            <Typography component="div">
              • <InlineMath math="\hat{\beta}" /> — market <Link href="/docs/capm-beta" passHref><MuiLink>beta</MuiLink></Link>
            </Typography>
            <Typography component="div">
              • <strong>t-stat / p-value</strong> on <InlineMath math="\alpha" /> — significance of skill
            </Typography>
          </Box>
        </Paper>
        
        {/* Backend Implementation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Our Backend Implementation
          </Typography>
          <Typography paragraph>
            Our portfolio optimization backend calculates Jensen's Alpha using the following approach:
          </Typography>
          
          <Box component="pre" sx={{ p: 2, bgcolor: '#f1f1f1', borderRadius: 1, overflow: 'auto', fontSize: '0.875rem' }}>
            <code>
              X       = sm.add_constant(bench_excess.values)   # [1, x_t]{'\n'}
              model   = sm.OLS(port_excess.values, X){'\n'}
              result  = model.fit(){'\n'}
              {'\n'}
              daily_alpha   = result.params[0]{'\n'}
              beta          = result.params[1]{'\n'}
              portfolio_alpha = daily_alpha * 252     # annualise{'\n'}
              beta_pvalue     = result.pvalues[1]{'\n'}
              r_squared       = result.rsquared
            </code>
          </Box>
          
          <Box sx={{ mt: 2 }}>
            <ul>
              <li>
                <Typography component="div">
                  <strong>Excess returns</strong> are calculated against a daily risk-free series
                </Typography>
              </li>
              <li>
                <Typography component="div">
                  Alpha is annualized by multiplying the daily intercept by <strong>252</strong> trading days
                </Typography>
              </li>
              <li>
                <Typography component="div">
                  Stored as <code>portfolio_alpha</code> in the API response, shown alongside β, p-value, and <InlineMath math="R^{2}" />
                </Typography>
              </li>
            </ul>
          </Box>
        </Paper>
        
        {/* Interpreting Jensen's Alpha */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Interpreting Jensen's Alpha
          </Typography>
          
          <TableContainer component={Paper} elevation={0} sx={{ mb: 3 }}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>α Value</strong></TableCell>
                  <TableCell><strong>Meaning</strong></TableCell>
                  <TableCell><strong>Practical Take-away</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell><InlineMath math="\alpha > 0" /></TableCell>
                  <TableCell>Strategy beat CAPM expectations</TableCell>
                  <TableCell>Indicates skill or unique edge</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><InlineMath math="\alpha = 0" /></TableCell>
                  <TableCell>Matches risk-adjusted benchmark</TableCell>
                  <TableCell>Pure β exposure—no evidence of skill</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><InlineMath math="\alpha < 0" /></TableCell>
                  <TableCell>Under-performed given its β</TableCell>
                  <TableCell>Destroyed value versus passive market</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
          
          <Typography paragraph>
            Always pair α with <strong>p-value</strong> or <strong>t-stat</strong> to confirm statistical significance. A high alpha that is not 
            statistically significant might be due to chance rather than skill.
          </Typography>
        </Paper>
        
        {/* Typical Use-Cases */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Typical Use-Cases
          </Typography>
          
          <TableContainer component={Paper} elevation={0}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Use-Case</strong></TableCell>
                  <TableCell><strong>How Alpha Helps</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell><strong>Fund selection</strong></TableCell>
                  <TableCell>Identify managers delivering true skill</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Performance fees</strong></TableCell>
                  <TableCell>Many hedge-fund agreements pay incentive fees only on positive α</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Style drift monitoring</strong></TableCell>
                  <TableCell>Persistent α turning negative suggests strategy degradation</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
        
        {/* Limitations & Best Practice */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Limitations & Best Practice
          </Typography>
          
          <TableContainer component={Paper} elevation={0}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Issue</strong></TableCell>
                  <TableCell><strong>Mitigation</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell><strong>Model mis-specification</strong></TableCell>
                  <TableCell>Use multi-factor models (Fama-French, Carhart) to separate size, value, momentum premiums</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Beta Instability</strong></TableCell>
                  <TableCell>Compute rolling α/β to detect regime changes</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Short sample noise</strong></TableCell>
                  <TableCell>Prefer ≥ 3 years monthly data or ≥ 250 daily observations for robust inference</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>

        {/* Advantages and Limitations Section */}
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
                      <strong>Benchmark-adjusted assessment:</strong> Evaluates performance specifically relative to a relevant benchmark, not just in absolute terms.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Risk-adjustment:</strong> Accounts for the level of systematic risk taken, enabling fair comparison between strategies with different risk levels.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Statistical validation:</strong> Can be tested for statistical significance to determine if outperformance is likely skill-based or simply due to chance.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Skill identification:</strong> Provides a clear distinction between returns generated through manager skill versus those from general market exposure.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Academically robust:</strong> Based on established financial theory and supported by decades of empirical research in portfolio performance evaluation.
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
                      <strong>CAPM dependency:</strong> Inherits all limitations of the CAPM model, including assumptions about market efficiency and investor rationality.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Benchmark sensitivity:</strong> Results can vary dramatically based on which benchmark is chosen as the market proxy.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Single-factor limitation:</strong> Ignores other systematic risk factors that might explain returns beyond market risk alone.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Time period dependency:</strong> Alpha values can be highly sensitive to the specific time period used in the analysis.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Data requirements:</strong> Needs sufficient historical data to produce statistically meaningful results, potentially limiting usefulness for new strategies.
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
              <Typography component="div">
                <strong>Jensen, M. C. (1968)</strong>. "The Performance of Mutual Funds in the Period 1945–1964." <em>Journal of Finance</em>, 23(2), 389-416.
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <strong>Roll, R. (1978)</strong>. "Ambiguity when Performance Is Measured by the Securities Market Line." <em>Journal of Finance</em>, 33(4), 1051-1069.
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <strong>Bodie, Kane & Marcus</strong>. <em>Investments</em> (12 ed.), McGraw-Hill, 2021 — Ch. 24 (Performance Evaluation).
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
                  Capital Asset Pricing Model (CAPM)
                </Typography>
                <Typography variant="body2" paragraph>
                  The foundational theory behind alpha and beta, explaining the relationship between systematic risk and expected return.
                </Typography>
                <Link href="/docs/capm" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  CAPM Beta (β)
                </Typography>
                <Typography variant="body2" paragraph>
                  A measure of systematic risk that represents how an asset moves relative to the overall market.
                </Typography>
                <Link href="/docs/capm-beta" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Sharpe Ratio
                </Typography>
                <Typography variant="body2" paragraph>
                  A measure of risk-adjusted return that helps investors understand the return of an investment compared to its risk.
                </Typography>
                <Link href="/docs/sharpe-ratio" passHref>
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

export default JensensAlphaPage; 