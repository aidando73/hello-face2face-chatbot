import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import torchaudio.compliance.kaldi as ta_kaldi
import numpy as np
from audio_encoder import AudioEncoder
import os

class MelFilterBank:
    def __init__(
        self,
        sampling_rate: int = 16000,
        num_mel_bins: int = 80,
    ):
        self.sampling_rate = sampling_rate
        self.num_mel_bins = num_mel_bins
        
    def process_audio(self, audio_path: str) -> torch.Tensor:
        # Load audio file
        if os.environ.get("DEBUG"):
            print(f"Loading audio file: {audio_path}")
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Debug: Print waveform stats
        if os.environ.get("DEBUG"):
            print(f"Waveform stats - mean: {waveform.mean().item():.4f}, std: {waveform.std().item():.4f}, min: {waveform.min().item():.4f}, max: {waveform.max().item():.4f}")
        
        # Resample if necessary
        if sample_rate != self.sampling_rate:
            resampler = T.Resample(sample_rate, self.sampling_rate)
            waveform = resampler(waveform)
            print(f"Resampling from {sample_rate} to {self.sampling_rate}")
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            print(f"Converted to mono: {waveform.shape}")
        
        # Debug: Print resampled waveform stats
        if os.environ.get("DEBUG"):
            print(f"Resampled waveform stats - mean: {waveform.mean().item():.4f}, std: {waveform.std().item():.4f}, min: {waveform.min().item():.4f}, max: {waveform.max().item():.4f}")
        

        waveform = waveform * (2**15) # Kaldi compliance: 16-bit signed integers
        # Compute mel spectrogram
        mel_spec = ta_kaldi.fbank(
            waveform,
            num_mel_bins=self.num_mel_bins,
            sample_frequency=self.sampling_rate,
            frame_length=25,
            frame_shift=10,
            dither=1.0,
            energy_floor=0.0,
        )

        # Transpose the mel spectrogram to match the expected shape [1, 80, T]
        # Current shape is [T, 80], where T is the time dimension (1037 in this example)
        mel_spec = mel_spec.transpose(0, 1)
        mel_spec = mel_spec.unsqueeze(0)
        print(f"Mel spec shape: {mel_spec.shape}")

        # Print statistics about the mel spectrogram
        if os.environ.get("DEBUG"):
            print(f"Pre-normalized mel spectrogram statistics:")
            print(f"  Shape: {mel_spec.shape}")
            print(f"  Mean: {mel_spec.mean().item():.4f}")
            print(f"  Std: {mel_spec.std().item():.4f}")
            print(f"  Min: {mel_spec.min().item():.4f}")
            print(f"  Max: {mel_spec.max().item():.4f}")
            print(f"  Non-zero elements: {torch.count_nonzero(mel_spec).item()} ({torch.count_nonzero(mel_spec).item() / mel_spec.numel() * 100:.2f}%)")

        mel_spec = self.normalize(mel_spec)

        # Print statistics about the normalized mel spectrogram
        if os.environ.get("DEBUG"):
            print(f"Normalized mel spectrogram statistics:")
            print(f"  Shape: {mel_spec.shape}")
            print(f"  Mean: {mel_spec.mean().item():.4f}")
            print(f"  Std: {mel_spec.std().item():.4f}")
            print(f"  Min: {mel_spec.min().item():.4f}")
            print(f"  Max: {mel_spec.max().item():.4f}")
        return mel_spec
    
    def visualize(self, mel_spec: torch.Tensor, save_path: str = None):
        # Convert to numpy for visualization
        mel_spec_np = mel_spec.squeeze().numpy()
        
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_spec_np, aspect='auto', origin='lower')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        plt.xlabel('Time')
        plt.ylabel('Mel Frequency')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def normalize(self, mel_spec: torch.Tensor) -> torch.Tensor:
        cmvn_means = [
            11.837255115918403,
            12.473204615847946,
            13.416767619583318,
            14.077409846458519,
            14.692713667734644,
            15.134646755356338,
            15.425053998320841,
            15.520304088736482,
            15.664980906057181,
            15.682885368714361,
            15.83134095973795,
            15.901056812316575,
            16.043105914428832,
            16.141928413478638,
            16.146063740161384,
            16.17268368755442,
            16.13231180127601,
            16.065540090344545,
            16.170683092860383,
            15.998216926090535,
            15.867837768614727,
            16.081028935225024,
            15.90913485828459,
            16.032066529724602,
            15.94857810175373,
            16.03539817911192,
            15.919972463810511,
            16.012130517613077,
            15.93573072975294,
            15.914797286475908,
            15.949416173227279,
            15.914241247262952,
            15.9205949984345,
            15.979177455555364,
            15.986889776762691,
            16.04603056604172,
            16.110854420018935,
            16.11681722403251,
            16.129875546992444,
            16.085759281189265,
            16.134709075491045,
            16.09818475127177,
            16.202892094198077,
            16.195676195628295,
            16.265984774543206,
            16.368600951439756,
            16.48524192770144,
            16.53072364237602,
            16.58613266332892,
            16.682058026108336,
            16.643586991407417,
            16.62329213337083,
            16.638263919106894,
            16.703993486441295,
            16.75845666749587,
            16.818435528248443,
            16.88729840520967,
            16.89038585593233,
            16.816687157527294,
            16.731004380992307,
            16.674947603018126,
            16.562815703508104,
            16.50694580056838,
            16.427151307327705,
            16.33695716109585,
            16.22435176840036,
            16.122595445956836,
            16.074572001519112,
            16.045862034568927,
            15.997705599309137,
            15.955502796282088,
            15.925529416522258,
            15.884868619147634,
            15.847951054825177,
            15.812488364237238,
            15.791251105720136,
            15.698867196814575,
            15.451057143452907,
            15.043111236177761,
            14.453490694177178
        ]
        cmvn_stds = [
            0.3329853392031094,
            0.304457074540202,
            0.29546332871219255,
            0.3016748868710893,
            0.30146666620931134,
            0.29722031038819924,
            0.2917111074466677,
            0.28384523520067434,
            0.28523771805804266,
            0.2901033423245173,
            0.29489059636859316,
            0.29547348893246006,
            0.29620672776488977,
            0.2956320049313563,
            0.29470081603763376,
            0.2963227174569345,
            0.2976606657847016,
            0.3003491761055444,
            0.30211458429446625,
            0.30174903334867953,
            0.30016818970986625,
            0.3033775994936976,
            0.30356758993511046,
            0.30603640492289896,
            0.30671985841954447,
            0.3069228273662989,
            0.30770085196779645,
            0.3067656201661381,
            0.3055203785780098,
            0.30690623983421333,
            0.30723413937044297,
            0.3088911065771803,
            0.3091251382267279,
            0.30986769010126597,
            0.31000059868830204,
            0.30963732259143195,
            0.3093967488140671,
            0.30918562772813507,
            0.30968744324817,
            0.3085437993502015,
            0.309308051573859,
            0.3087313674687873,
            0.30814804868295664,
            0.30722053416625006,
            0.30732656820194293,
            0.3064066246045986,
            0.30390658225471334,
            0.302131011830547,
            0.3014575331911756,
            0.301449202764865,
            0.30039048343978525,
            0.29975195531574894,
            0.2993214974016792,
            0.29809597194189,
            0.2950458103872353,
            0.29250998818879875,
            0.29285432953044965,
            0.2928594451679315,
            0.2922642564293608,
            0.2934287968886421,
            0.2929937863211079,
            0.2921845930953747,
            0.2917417094543235,
            0.28991734472060865,
            0.2888153105794442,
            0.2870270977983177,
            0.2843282542200158,
            0.2827033299131669,
            0.28035104778082265,
            0.2782082983359874,
            0.27589950120001683,
            0.27325842201376005,
            0.27104919439201897,
            0.2688075817805597,
            0.26814315263564775,
            0.26998556725462286,
            0.269346874312791,
            0.2673887565870066,
            0.2683233739448121,
            0.2702135698992237
        ]
        # Convert means list to tensor for subtraction
        cmvn_means_tensor = torch.tensor(cmvn_means, dtype=mel_spec.dtype, device=mel_spec.device)
        # Reshape means tensor to match mel_spec dimensions for broadcasting
        # mel_spec shape is [1, num_mel_bins, time_steps]
        cmvn_means_tensor = cmvn_means_tensor.view(1, -1, 1)
        # Subtract mean on a per-feature basis (across the frequency dimension)
        mel_spec = mel_spec - cmvn_means_tensor

        # Convert stds list to tensor for division
        cmvn_stds_tensor = torch.tensor(cmvn_stds, dtype=mel_spec.dtype, device=mel_spec.device)
        # Reshape stds tensor to match mel_spec dimensions for broadcasting
        cmvn_stds_tensor = cmvn_stds_tensor.view(1, -1, 1)
        # Divide by standard deviation on a per-feature basis (across the frequency dimension)
        mel_spec = mel_spec * cmvn_stds_tensor
        
        return mel_spec

def main():
    # Initialize mel filter bank
    mel_filter = MelFilterBank()
    
    # Process audio file
    mel_spec = mel_filter.process_audio('audio-sample.wav')
    
    # Print mel spectrogram information
    if os.environ.get("DEBUG"): 
        print("\nMel Spectrogram:")
        print(f"Shape: {mel_spec.shape}")
        print(f"Mean: {mel_spec.mean():.4f}")
        print(f"Std: {mel_spec.std():.4f}")
        print(f"Min: {mel_spec.min():.4f}")
        print(f"Max: {mel_spec.max():.4f}")
    
    # Visualize mel spectrogram
    mel_filter.visualize(mel_spec, 'mel_spectrogram.png')


if __name__ == "__main__":
    main()
