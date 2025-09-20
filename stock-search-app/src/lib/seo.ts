export const SITE_URL =
  process.env.NEXT_PUBLIC_SITE_URL ||
  process.env.SITE_URL ||
  'https://indportfoliooptimization.vercel.app';

export function canonical(pathname: string) {
  if (!pathname || pathname === '/') return `${SITE_URL}/`;
  return `${SITE_URL}${pathname.startsWith('/') ? '' : '/'}${pathname}`;
}

export type SEOOpts = {
  title: string;
  description: string;
  path?: string;
  image?: string; // absolute or /relative under /public
};

export function ogImage(url?: string) {
  if (!url) return `${SITE_URL}/og/default.png`;
  if (url.startsWith('http')) return url;
  return `${SITE_URL}${url.startsWith('/') ? '' : '/'}${url}`;
}
