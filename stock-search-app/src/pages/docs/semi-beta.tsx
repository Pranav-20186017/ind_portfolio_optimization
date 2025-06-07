import React from "react";
import {
  Typography,
  Container,
  Box,
  Paper,
  Grid,
  Link as MuiLink,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Button,
  Card,
  CardContent,
  Divider
} from "@mui/material";
import Head from "next/head";
import Link from "next/link";
import "katex/dist/katex.min.css";
import { BlockMath, InlineMath } from "react-katex";
import TopNav from "../../components/TopNav";

/* -------------------------------------------------------------------
   EQUATION COMPONENT                                                    
   ------------------------------------------------------------------- */
const Equation: React.FC<{ math: string }> = ({ math }) => (
  <Box sx={{ p: 2, bgcolor: "#f5f5f5", borderRadius: 1, my: 2, textAlign: "center" }}>
    <BlockMath math={math} />
  </Box>
);

/* -------------------------------------------------------------------
   SEMI BETA PAGE                                                      
   ------------------------------------------------------------------- */
const SemiBetaPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Semi Beta (Downside Beta) | Portfolio Optimisation</title>
        <meta
          name="description"
          content="A downside risk measure that captures asset sensitivity to negative market movements, providing a more nuanced view of risk during market downturns."
        />
      </Head>

      <TopNav />

      <Container maxWidth="lg" sx={{ py: 6 }}>
        {/* NAVIGATION */}
        <Box sx={{ mb: 4, display: 'flex', gap: 2 }}>
          <Link href="/docs" passHref>
            <Button variant="outlined" color="primary">← Back to Docs</Button>
          </Link>
          <Link href="/" passHref>
            <Button variant="outlined" color="secondary">← Back to Portfolio Optimizer</Button>
          </Link>
        </Box>

        {/* HEADER */}
        <Box sx={{ mb: 6, textAlign: 'center' }}>
          <Typography variant="h3" component="h1" gutterBottom>
            Semi Beta (Downside Beta)
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            A risk measure that isolates an asset's sensitivity to negative market returns, offering better insight into downside protection.
          </Typography>
        </Box>

        {/* INTUITION */}
        <Paper elevation={3} sx={{ p: 3, my: 4 }}>
          <Typography variant="h5" gutterBottom>
            Why Semi Beta?
          </Typography>
          <Typography paragraph>
            Traditional beta treats upside and downside market movements equally, but investors are typically more concerned 
            with losses than gains. <strong>Semi Beta</strong> addresses this asymmetry by measuring an asset's sensitivity 
            specifically to negative market returns, providing crucial information about how securities behave during market downturns.
          </Typography>
          <Typography paragraph>
            Assets with lower semi-beta values offer better downside protection, making them valuable for defensive portfolio construction 
            and risk management during bear markets or market crashes.
          </Typography>
        </Paper>

        {/* MATHEMATICS */}
        <Typography variant="h4" gutterBottom>
          1. Mathematical Definition
        </Typography>
        <Typography variant="h6" gutterBottom>
          1.1 The Asymmetry Problem
        </Typography>
        <Typography paragraph>
          Traditional beta coefficient assumes that asset returns have a symmetric relationship with market returns:
        </Typography>
        <Equation math="\beta = \frac{\operatorname{Cov}(r_i,\,r_m)}{\operatorname{Var}(r_m)}" />
        <Typography paragraph>
          However, empirical evidence shows that many assets respond differently to positive versus negative market movements. 
          Semi-beta captures this asymmetry by conditioning the analysis on the sign of market returns.
        </Typography>

        <Typography variant="h6" gutterBottom>
          1.2 Downside Beta Calculation
        </Typography>
        <Typography paragraph>
          Semi Beta (Downside Beta) focuses exclusively on periods when the market return is below a threshold <InlineMath math="\tau" />, 
          typically set to zero:
        </Typography>
        <Equation math="\beta^- = \frac{\operatorname{Cov}(r_i,\,r_m | r_m < \tau)}{\operatorname{Var}(r_m | r_m < \tau)}" />

        <Typography paragraph>
          In practical implementation, this can be computed using regression on a filtered dataset:
        </Typography>
        <Equation math="r_i = \alpha^- + \beta^- r_m + \epsilon, \quad \text{for all } r_m < \tau" />
        <Typography paragraph>
          Where <InlineMath math="r_i" /> is the asset return, <InlineMath math="r_m" /> is the market return, <InlineMath math="\tau" /> is the threshold 
          (usually 0), <InlineMath math="\alpha^-" /> is the downside alpha, <InlineMath math="\beta^-" /> is the downside beta, 
          and <InlineMath math="\epsilon" /> is the error term.
        </Typography>

        <Typography variant="h6" gutterBottom>
          1.3 Interpretation and Financial Significance
        </Typography>
        <Typography paragraph>
          Semi Beta can be interpreted as follows:
        </Typography>
        <ul>
          <li>
            <Typography paragraph>
              <InlineMath math="\beta^- > 1" />: The asset amplifies negative market movements, losing more than the market during downturns.
            </Typography>
          </li>
          <li>
            <Typography paragraph>
              <InlineMath math="\beta^- = 1" />: The asset moves in tandem with the market during downturns.
            </Typography>
          </li>
          <li>
            <Typography paragraph>
              <InlineMath math="\beta^- < 1" />: The asset provides some cushioning against market downturns.
            </Typography>
          </li>
          <li>
            <Typography paragraph>
              <InlineMath math="\beta^- \leq 0" />: The asset provides significant protection or moves opposite to the market during downturns.
            </Typography>
          </li>
        </ul>
        <Typography paragraph>
          The difference between standard beta and semi-beta (<InlineMath math="\beta - \beta^-" />) indicates asymmetry in market response. 
          When this gap is large, it signals that the asset behaves very differently in bull versus bear markets.
        </Typography>

        <Typography variant="h6" gutterBottom>
          1.4 Upside Beta (For Comparison)
        </Typography>
        <Typography paragraph>
          Symmetrically, Upside Beta focuses on periods when the market return exceeds the threshold:
        </Typography>
        <Equation math="\beta^+ = \frac{\operatorname{Cov}(r_i,\,r_m | r_m > \tau)}{\operatorname{Var}(r_m | r_m > \tau)}" />
        <Typography paragraph>
          The relationship between standard beta, downside beta, and upside beta provides valuable insights into an asset's 
          complete risk profile across different market conditions.
        </Typography>

        {/* DEFAULT PARAMS */}
        <Typography variant="h4" gutterBottom>
          2. Default Parameters
        </Typography>
        <TableContainer component={Paper} sx={{ mb: 4 }}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Parameter</TableCell>
                <TableCell>Default</TableCell>
                <TableCell>Description</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {[
                ["threshold", "0.0", "Market return threshold (τ) for defining downside events"],
                ["window", "252", "Rolling window length in trading days (~1 year)"],
                ["min_periods", "60", "Minimum number of downside observations required"],
                ["rf", "0.0", "Risk-free rate subtracted from returns before estimation"],
              ].map(([param, def, desc]) => (
                <TableRow key={param as string}>
                  <TableCell>{param}</TableCell>
                  <TableCell>
                    <code>{def as string}</code>
                  </TableCell>
                  <TableCell>{desc}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>

        {/* IMPLEMENTATION CONSIDERATIONS */}
        <Typography variant="h4" gutterBottom>
          3. Implementation Considerations
        </Typography>
        <Typography paragraph>
          When implementing Semi Beta in portfolio optimization:
        </Typography>
        <ul>
          <li>
            <Typography paragraph>
              <strong>Sample size concerns:</strong> Filtering for downside market movements reduces the number of observations, 
              potentially leading to less reliable estimates. Ensure sufficient data points (typically 60+) for statistical validity.
            </Typography>
          </li>
          <li>
            <Typography>
              <strong>Threshold selection:</strong> While zero is the most common threshold, alternative values can be used:
            </Typography>
            <Box component="div" sx={{ pl: 2, mt: 1, mb: 2 }}>
              <Typography>• Zero (absolute): <InlineMath math="\tau = 0" /></Typography>
              <Typography>• Risk-free rate: <InlineMath math="\tau = r_f" /></Typography>
              <Typography>• Market mean: <InlineMath math="\tau = \mathbb{E}[r_m]" /></Typography>
            </Box>
          </li>
          <li>
            <Typography paragraph>
              <strong>Window length trade-off:</strong> Longer windows provide more downside observations but may include outdated information. 
              Shorter windows are more responsive to regime changes but may contain insufficient downside events.
            </Typography>
          </li>
          <li>
            <Typography paragraph>
              <strong>Complementary metrics:</strong> Consider calculating both downside and upside betas to fully understand asymmetric risk behavior.
            </Typography>
          </li>
        </ul>

        {/* PROS / CONS */}
        <Typography variant="h4" gutterBottom>
          4. Advantages and Limitations
        </Typography>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
              <Typography variant="h6" gutterBottom color="primary">
                Advantages
              </Typography>
              <ul>
                <li>Isolates sensitivity to negative market returns, aligning with investor risk aversion.</li>
                <li>Better identifies defensive assets that provide downside protection.</li>
                <li>Captures asymmetric market response not reflected in standard beta.</li>
                <li>More relevant for risk management during market crises.</li>
                <li>Complements standard beta for comprehensive risk assessment.</li>
              </ul>
            </Box>
          </Grid>
          <Grid item xs={12} md={6}>
            <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
              <Typography variant="h6" gutterBottom color="error">
                Limitations
              </Typography>
              <ul>
                <li>Requires sufficient market downturns for statistical reliability.</li>
                <li>More sensitive to the estimation window than standard beta.</li>
                <li>Threshold selection (<InlineMath math="\tau" />) introduces subjectivity.</li>
                <li>May yield unstable estimates in prolonged bull markets with few downside observations.</li>
                <li>Not directly incorporated in most standard asset pricing models.</li>
              </ul>
            </Box>
          </Grid>
        </Grid>

        {/* REFERENCES */}
        <Paper elevation={2} sx={{ p: 4, mb: 4, mt: 4 }}>
          <Typography variant="h4" gutterBottom>
            5. References
          </Typography>
          <ul>
            <li>Bawa, V. S., & Lindenberg, E. B. (1977). <em>Capital Market Equilibrium in a Mean-Lower Partial Moment Framework</em>. <em>Journal of Financial Economics</em>, 5(2), 189-200.</li>
            <li>Ang, A., Chen, J., & Xing, Y. (2006). <em>Downside Risk</em>. <em>Review of Financial Studies</em>, 19(4), 1191-1239.</li>
            <li>Estrada, J. (2002). <em>Systematic Risk in Emerging Markets: The D-CAPM</em>. <em>Emerging Markets Review</em>, 3(4), 365-379.</li>
            <li>Levi, Y., & Welch, I. (2020). <em>Symmetric and Asymmetric Market Betas and Downside Risk</em>. <em>Review of Financial Studies</em>, 33(6), 2772–2795.</li>
            <li>Post, T., & Van Vliet, P. (2006). <em>Downside Risk and Asset Pricing</em>. <em>Journal of Banking & Finance</em>, 30(3), 823-849.</li>
          </ul>
        </Paper>

        {/* RELATED TOPICS */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h4" gutterBottom>
            Related Topics
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={3}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Welch Beta
                </Typography>
                <Typography variant="body2" paragraph>
                  A robust alternative to traditional beta that uses winsorization to reduce the impact of extreme returns.
                </Typography>
                <Link href="/docs/welch-beta" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Portfolio Beta
                </Typography>
                <Typography variant="body2" paragraph>
                  Traditional measure of systematic risk that represents how an asset moves relative to the overall market.
                </Typography>
                <Link href="/docs/capm-beta" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Conditional Value-at-Risk
                </Typography>
                <Typography variant="body2" paragraph>
                  A risk measure that quantifies the expected loss in the worst-case scenarios beyond the VaR threshold.
                </Typography>
                <Link href="/docs/conditional-value-at-risk" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Sortino Ratio
                </Typography>
                <Typography variant="body2" paragraph>
                  A modification of the Sharpe ratio that only penalizes returns falling below a specified target.
                </Typography>
                <Link href="/docs/sortino-ratio" passHref>
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

export default SemiBetaPage; 