import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
from typing import Dict
import warnings
warnings.filterwarnings('ignore')


class GasPipelineRiskModel:
    """
    Main class for the gas pipeline risk model
    """

    def __init__(self):
        self.theta_columns = [
            'Pipe_Age', 'Pipe_Material', 'Burial_Depth', 'Pressure',
            'Moisture', 'Slope', 'Distance_to_Road', 'Urban_or_Rural'
        ]

    def load_data(self, economic_path: str, technical_path: str, probability_path: str) -> Dict:
        """
        Load all required datasets from Excel files. Returns a dict with keys:
        'economic', 'technical', 'probability'.
        """
        try:
            data = {}
            data['economic'] = pd.read_excel(economic_path)
            print(f"✅ Economic data loaded: {len(data['economic'])} rows")
            data['technical'] = pd.read_excel(technical_path)
            print(f"✅ Technical data loaded: {len(data['technical'])} rows")
            data['probability'] = pd.read_excel(probability_path)
            print(f"✅ Probability data loaded: {len(data['probability'])} rows")
            return data
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return {}

    def calculate_direct_cost(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate direct economic cost.
        Formula:
          (Houses_200m * Damage_Severity * UnitCost_House) +
          (Vehicles_ADT * Time_Weight * UnitCost_Vehicle) +
          (Road_Length_m * Damage_Severity * UnitCost_Road_m)
        """
        df = df.copy()
        df['Direct_Cost_Dollar'] = (
            df['Houses_200m'] * df['Damage_Severity'] * df['UnitCost_House'] +
            df['Vehicles_ADT'] * df['Time_Weight'] * df['UnitCost_Vehicle'] +
            df['Road_Length_m'] * df['Damage_Severity'] * df['UnitCost_Road_m']
        )
        return df

    def calculate_technical_cost(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical cost.
        Formula:
          (Time_Weight * theta_f) * (
               Gas_Interruption_Hour * UnitCost_Outage_perHour +
               Critical_Equipments_Count * UnitCost_Critical_Equipment +
               SCADA_Sensors_Count * UnitCost_SCADA_Sensor
          )
        """
        df = df.copy()
        # theta factor column is expected as 'theta_f' or 'θ_f' in input; accept both if present
        if 'θ_f' in df.columns and 'theta_f' not in df.columns:
            df['theta_f'] = df['θ_f']

        df['Technical_Cost_$'] = (
            df['Time_Weight'] * df['theta_f'] * (
                df['Gas_Interruption_Hour'] * df['UnitCost_Outage_perHour'] +
                df['Critical_Equipments_Count'] * df['UnitCost_Critical_Equipment'] +
                df['SCADA_Sensors_Count'] * df['UnitCost_SCADA_Sensor']
            )
        )
        return df

    def survival_function(self, lambda_rain: float, R_rain: float,
                          lambda_snow: float, R_snow: float) -> float:
        """
        Survival function F(theta) = exp(-(lambda_rain * R_rain + lambda_snow * R_snow))
        Returns a multiplicative factor in [0,1].
        """
        return np.exp(-(lambda_rain * R_rain + lambda_snow * R_snow))

    def shannon_entropy(self, p: float, eps: float = 1e-12) -> float:
        """
        Compute Shannon entropy for uncertainty weighting of a probability p.
        Values near 0.5 yield higher entropy; values near 0 or 1 yield lower entropy.
        """
        p_clip = np.clip(p, eps, 1 - eps)
        return -(p_clip * np.log(p_clip) + (1 - p_clip) * np.log(1 - p_clip))

    def calculate_total_risk(self, data: Dict, lambda_rain: float = 0.1,
                             lambda_snow: float = 0.05) -> pd.DataFrame:
        """
        Compute final risk for each merged record.
        Risk per record is computed as:
          Risk = P_explosion * C_total * Time_Weight * Shannon_Entropy(P_explosion) * F_theta
        where F_theta is the survival function value.
        """
        merged_data = self.merge_datasets(data)
        risk_results = []

        for idx, row in merged_data.iterrows():
            P_explosion = row['Explosion_Probability']
            C_direct = row['Direct_Cost_Dollar']
            C_technical = row['Technical_Cost_$']
            C_total = C_direct + C_technical

            w_time = row.get('Time_Weight', 1.0)
            w_entropy = self.shannon_entropy(P_explosion)
            # default R_rain=1.0 and R_snow=0.5 are placeholders; adapt to real data if available
            F_theta = self.survival_function(lambda_rain, 1.0, lambda_snow, 0.5)

            risk = P_explosion * C_total * w_time * w_entropy * F_theta

            risk_results.append({
                'Zone': row['Zone'],
                'Time': row['Time'],
                'Explosion_Probability': P_explosion,
                'Direct_Cost': C_direct,
                'Technical_Cost': C_technical,
                'Total_Cost': C_total,
                'Time_Weight': w_time,
                'Shannon_Entropy': w_entropy,
                'Survival_Function': F_theta,
                'Final_Risk': risk
            })

        return pd.DataFrame(risk_results)

    def merge_datasets(self, data: Dict) -> pd.DataFrame:
        """
        Merge economic, technical and probability datasets on ['Zone', 'Time']
        """
        merged = pd.merge(data['economic'], data['technical'], on=['Zone', 'Time'], how='inner')
        merged = pd.merge(merged, data['probability'], on=['Zone', 'Time'], how='inner')
        return merged

    def generate_risk_report(self, risk_df: pd.DataFrame) -> Dict:
        """
        Generate a dictionary with summary statistics and analyses:
        - total, average, max, min risk
        - per-zone aggregation
        - per-time aggregation
        - correlation matrix between key columns
        """
        report = {}
        report['total_risk'] = risk_df['Final_Risk'].sum()
        report['average_risk'] = risk_df['Final_Risk'].mean()
        report['max_risk'] = risk_df['Final_Risk'].max()
        report['min_risk'] = risk_df['Final_Risk'].min()

        zone_risk = risk_df.groupby('Zone')['Final_Risk'].agg(['sum', 'mean', 'std']).round(2)
        report['zone_analysis'] = zone_risk

        time_risk = risk_df.groupby('Time')['Final_Risk'].agg(['sum', 'mean', 'std']).round(2)
        report['time_analysis'] = time_risk

        correlation = risk_df[['Explosion_Probability', 'Total_Cost', 'Final_Risk']].corr()
        report['correlation_analysis'] = correlation

        return report

    def plot_risk_analysis(self, risk_df: pd.DataFrame, save_path: str = None):
        """
        Produce a set of plots to visualize the risk analysis.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Gas Pipeline Risk Analysis', fontsize=16, fontweight='bold')

        axes[0, 0].hist(risk_df['Final_Risk'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Final Risk Distribution')
        axes[0, 0].set_xlabel('Risk Value')
        axes[0, 0].set_ylabel('Count')

        zone_risk = risk_df.groupby('Zone')['Final_Risk'].mean()
        axes[0, 1].bar(zone_risk.index, zone_risk.values, alpha=0.7)
        axes[0, 1].set_title('Average Risk by Zone')
        axes[0, 1].set_xlabel('Zone')
        axes[0, 1].set_ylabel('Average Risk')

        time_risk = risk_df.groupby('Time')['Final_Risk'].mean()
        axes[0, 2].bar(time_risk.index, time_risk.values, alpha=0.7)
        axes[0, 2].set_title('Average Risk by Time')
        axes[0, 2].set_xlabel('Time')
        axes[0, 2].set_ylabel('Average Risk')

        axes[1, 0].scatter(risk_df['Explosion_Probability'], risk_df['Total_Cost'], alpha=0.6)
        axes[1, 0].set_title('Explosion Probability vs Total Cost')
        axes[1, 0].set_xlabel('Explosion Probability')
        axes[1, 0].set_ylabel('Total Cost ($)')

        cost_components = ['Direct_Cost', 'Technical_Cost']
        cost_sums = [risk_df['Direct_Cost'].sum(), risk_df['Technical_Cost'].sum()]
        axes[1, 1].pie(cost_sums, labels=cost_components, autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Share of Cost Components')

        axes[1, 2].scatter(risk_df['Shannon_Entropy'], risk_df['Final_Risk'], alpha=0.6)
        axes[1, 2].set_title('Shannon Entropy vs Final Risk')
        axes[1, 2].set_xlabel('Shannon Entropy')
        axes[1, 2].set_ylabel('Final Risk')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Charts saved to {save_path}")

        plt.show()


def main():
    """
    Main execution function
    """
    print("Starting Gas Pipeline Risk Assessment Model")
    print("=" * 50)

    model = GasPipelineRiskModel()

    # Update these file paths to point to your Excel files
    data_paths = {
        'economic': 'C_economic_54rows_WITH_FORMULA.xlsx',
        'technical': 'technical_cost_risk_model_final_WITH_FORMULA.xlsx',
        'probability': 'risk_model_theta8_with_time_weight.xlsx'
    }

    data = model.load_data(**data_paths)

    if not data:
        print("Error loading data. Exiting.")
        return

    print("\nCalculating costs...")
    data['economic'] = model.calculate_direct_cost(data['economic'])
    data['technical'] = model.calculate_technical_cost(data['technical'])

    print("Calculating final risk...")
    risk_results = model.calculate_total_risk(data)

    print("Generating report...")
    report = model.generate_risk_report(risk_results)

    print("\n" + "=" * 50)
    print("Risk Summary")
    print("=" * 50)
    print(f"Total Risk: {report['total_risk']:,.2f} $")
    print(f"Average Risk: {report['average_risk']:,.2f} $")
    print(f"Maximum Risk: {report['max_risk']:,.2f} $")
    print(f"Minimum Risk: {report['min_risk']:,.2f} $")

    print("\nRegional analysis:")
    print(report['zone_analysis'])

    print("\nProducing charts...")
    model.plot_risk_analysis(risk_results, 'risk_analysis_results.png')

    risk_results.to_excel('final_risk_assessment_results.xlsx', index=False)
    print("Final results saved to 'final_risk_assessment_results.xlsx'")

    print("\nRisk modeling completed successfully!")


if __name__ == '__main__':
    main()
