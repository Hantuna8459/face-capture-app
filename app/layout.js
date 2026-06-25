import "./globals.css";

export const metadata = {
  title: "Face Motion Recorder",
  description: "Record face motion from a laptop camera and save it locally.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
